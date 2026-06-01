"""Replicant Character Lab — a Wan2GP plugin.

A guided 7-step wizard (Info → Prompt → Base → Face/Body → Poses → Save → Train)
for building a reusable character and its LoRA datasets, ported from
SupremeDiffusion's PySide6 character creator into Wan2GP's Gradio UI.

NOTE: not an official plugin. Distribute via the plugin-manager "add from GitHub
URL" flow; do not add to the bundled plugins.json without dbm's approval.
"""
import traceback

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

try:  # GPU arbitration with the main Video Generator (see wan2gp-sample)
    from shared.utils.process_locks import (acquire_GPU_ressources,
                                            any_GPU_process_running,
                                            release_GPU_ressources)
    _HAVE_LOCKS = True
except Exception:  # pragma: no cover
    _HAVE_LOCKS = False

from .core import character, discovery, faceswap, gen_sd, models, paths, poses
from .ui import wizard
from .ui.styles import CSS

PLUGIN_ID = "ReplicantCharLab"
PLUGIN_NAME = "Replicant Character Lab"


class ReplicantCharLab(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PLUGIN_NAME
        self.version = "0.1.0"
        self.description = ("Character creator wizard: prompts, base image, poses, "
                            "LoRA datasets and training in a guided 7-step tab.")

    # -- lifecycle ----------------------------------------------------------
    def setup_ui(self):
        # First install / every boot: make sure the data dirs exist.
        try:
            paths.ensure_dirs()
        except Exception:
            traceback.print_exc()

        self.request_component("state")
        self.request_component("main_tabs")
        self.request_component("refresh_form_trigger")
        self.request_global("get_current_model_settings")
        # Native prompt enhancer (Qwen3.5 when enhancer_enabled in {3,4}).
        # exec_prompt_enhancer_engine self-manages the GPU lock + model load/unload.
        self.request_global("exec_prompt_enhancer_engine")
        self.request_global("get_state_model_type")
        self.request_global("get_model_def")
        self.request_global("get_default_settings")  # base/pose image generation
        self.request_global("models_def")             # list native image models

        self.add_tab(tab_id=PLUGIN_ID, label=PLUGIN_NAME,
                     component_constructor=self.create_ui)

    def on_tab_select(self, state: dict):
        """Block entering the lab while a Wan2GP generation is running: warn and
        bounce back to the Video Generator tab."""
        if _HAVE_LOCKS and any_GPU_process_running(state, PLUGIN_ID):
            gr.Warning("A generation is running — finish or stop it before using "
                       "the Character Lab.")
            try:
                return self.goto_video_tab(state)  # injected like in wan2gp-sample
            except Exception:
                return gr.update()
        return gr.update()

    # -- GPU arbitration helpers (used by step actions, wired later) --------
    def acquire_gpu(self, state):
        if not _HAVE_LOCKS:
            return True
        if any_GPU_process_running(state, PLUGIN_ID):
            gr.Error("Another process is using the GPU")
            return False
        acquire_GPU_ressources(state, PLUGIN_ID, PLUGIN_NAME, gr=gr)
        return True

    def release_gpu(self, state):
        if _HAVE_LOCKS:
            release_GPU_ressources(state, PLUGIN_ID)

    @staticmethod
    def _require(keys, what):
        """Gate a GPU action on explicitly-downloaded models — never auto-pull."""
        miss = models.missing(keys)
        if miss:
            raise gr.Error(f"{what} needs models you haven't downloaded yet — get them "
                           f"in Prerequisites → Models first: " + ", ".join(miss))

    # -- UI -----------------------------------------------------------------
    def _native_model_types(self):
        defs = getattr(self, "models_def", None) or {}
        try:
            return [mt for mt in defs if discovery.categorize_native(mt)]
        except Exception:
            return []

    def create_ui(self, api_session):
        self._api = api_session
        self._faceswap = None  # lazy FaceSwapPipeline
        model_choices = discovery.build_model_choices(self._native_model_types())
        lora_choices = discovery.lora_choices()  # categorized; filtered by model below
        from .core import wizard_state
        init = wizard_state.load()
        # These two render nothing visible; hide their wrappers so they don't add
        # flex-gap space above the logo.
        gr.HTML(f"<style>{CSS}</style>", elem_classes="replicant-hidden")
        # Tag our main-webui tab button (Gradio gives no elem_id for it) so the
        # purple-border CSS above can target only our tab. gr.HTML sets innerHTML,
        # which does NOT execute <script> tags — so run via an <img onerror> hook,
        # which fires even when inserted that way.
        gr.HTML(
            "<img src=x style='display:none' onerror=\"(function(){"
            "var NAME=" + repr(PLUGIN_NAME) + ";"
            "function mark(){document.querySelectorAll("
            "'.tab-nav button,button[role=&quot;tab&quot;]').forEach(function(b){"
            "if(b.textContent.trim()===NAME)b.classList.add('replicant-tabbtn');});}"
            "mark();new MutationObserver(mark).observe(document.body,"
            "{childList:true,subtree:true});})()\">",
            elem_classes="replicant-hidden")
        with gr.Column(elem_id="replicant-root"):
            ui = wizard.build_wizard(model_choices=model_choices, lora_choices=lora_choices,
                                     init=init)
        self._wire_enhancer(ui)
        self._wire_generation(ui)
        self.on_tab_outputs = [self.main_tabs] if hasattr(self, "main_tabs") else None
        self._ui = ui
        return ui

    # -- generation backends ------------------------------------------------
    def _face_pipe(self):
        if self._faceswap is None:
            self._faceswap = faceswap.FaceSwapPipeline(str(paths.models_dir() / "face"))
        return self._faceswap

    def _release_faceswap(self):
        """Free the InsightFace + ONNX swap models from VRAM."""
        try:
            if getattr(self, "_faceswap", None) is not None:
                self._faceswap.release()
        except Exception:
            pass
        self._faceswap = None
        gen_sd._free_torch()

    def _gen_image(self, state, model_value, pos, neg, w, h, steps, cfg, seed,
                   sampler, scheduler, clip_skip):
        """One image via the routed backend. Returns list of saved paths."""
        backend, ident = discovery.parse_model_value(model_value)
        if backend == "native":
            settings = dict(self.get_default_settings(ident))
            settings.update({
                "model_type": ident, "image_mode": 1, "prompt": pos,
                "negative_prompt": neg or "", "resolution": f"{int(w)}x{int(h)}",
                "num_inference_steps": int(steps), "guidance_scale": float(cfg),
                "seed": int(seed), "video_length": 1, "batch_size": 1,
            })
            result = self._api.submit_task(settings).result()
            if result.success and result.generated_files:
                return list(result.generated_files)
            if result.errors:
                raise gr.Error(str(list(result.errors)[0]))
            return []
        if backend == "sd":
            if not self.acquire_gpu(state):
                return []
            try:
                return gen_sd.generate_txt2img(
                    ident, pos, neg, w, h, steps, cfg, seed,
                    sampler=sampler or "DPM++ 2M", scheduler=scheduler or "",
                    clip_skip=int(clip_skip))
            finally:
                self.release_gpu(state)
        raise gr.Error("Select a model from the dropdown first.")

    def _wire_generation(self, ui):
        c, s = ui["components"], ui["settings"]
        base, prm, swap, pose = c["base"], c["setup"], c["swap"], c["poses"]
        if not getattr(self, "_api", None):
            return

        import random as _rng
        SET = [s["model"], s["sampler"], s["scheduler"], s["steps"], s["cfg_scale"],
               s["clip_skip"], s["seed"], s["width"], s["height"]]

        # LoRAs are family-specific (Pony≠Illustrious≠SDXL) — filter to the model's family.
        def _loras_for(model_value):
            return gr.update(choices=discovery.lora_choices(
                family=discovery.model_family(model_value)), value=[])
        s["model"].change(_loras_for, inputs=[s["model"]], outputs=[s["loras"]])

        # -- base candidates --
        def _gen_base(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                      width, height, pos, neg, count, progress=gr.Progress()):
            if not (pos and pos.strip()):
                raise gr.Error("Build or enhance a positive prompt on step 2 first.")
            self._release_faceswap()  # base gen is pure SD — free InsightFace VRAM
            files, n = [], int(count)
            for i in range(n):
                sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                progress((i, n), desc=f"Base {i + 1}/{n}")
                files += self._gen_image(state, model, pos, neg, width, height,
                                         steps, cfg, sd, sampler, scheduler, clip_skip)
            if not files:
                raise gr.Error("Generation produced no images.")
            return files, files[0]

        # Generation populates candidates + the pick pointer ONLY — the base never
        # changes automatically; it's committed via "Use as Base" / "Revert".
        base["generate"].click(
            _gen_base,
            inputs=[self.state] + SET + [base["pos"], base["neg"], base["count"]],
            outputs=[base["candidates"], base["picked"]])

        def _pick(evt: gr.SelectData):
            v = evt.value
            if isinstance(v, dict):
                return v.get("image", {}).get("path") or v.get("path") or gr.update()
            return v if isinstance(v, str) else gr.update()
        # Clicking a candidate only records the selection (no base change, no reflow).
        base["candidates"].select(_pick, outputs=[base["picked"]])
        base["use_as_base"].click(lambda p: p or gr.update(), inputs=[base["picked"]],
                                  outputs=[base["selected_base"]])

        # Reimagine the reference (img2img) — SD-family only.
        def _reimagine(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                       width, height, pos, neg, denoise, count, ref_img,
                       progress=gr.Progress()):
            backend, ident = discovery.parse_model_value(model)
            if backend != "sd":
                raise gr.Error("Reimagine (img2img) needs an SDXL/Pony/Illustrious model.")
            if not ref_img:
                raise gr.Error("No reference image to reimagine (add one on step 1).")
            if not (pos and pos.strip()):
                raise gr.Error("Build or enhance a positive prompt on step 2 first.")
            self._release_faceswap()
            if not self.acquire_gpu(state):
                return gr.update(), gr.update()
            files, n = [], int(count)
            try:
                for i in range(n):
                    sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                    progress((i, n), desc=f"Reimagining {i + 1}/{n} (img2img)")
                    files += gen_sd.generate_img2img(
                        ident, ref_img, pos, neg, width, height, steps, cfg, sd,
                        denoise=float(denoise), sampler=sampler, scheduler=scheduler,
                        clip_skip=int(clip_skip))
            finally:
                self.release_gpu(state)
            if not files:
                raise gr.Error("img2img produced no images.")
            return files, files[0]

        base["reimagine"].click(
            _reimagine,
            inputs=[self.state] + SET + [base["pos"], base["neg"],
                                         base["denoise"], base["count"], base["ref_avatar"]],
            outputs=[base["candidates"], base["picked"]])

        # -- step 4: face swap onto the base (optional) --
        # --- Swap review flow: run -> preview, then Retry or Accept (commit to base).
        # While a result is pending, the other swap + step navigation lock so a bad
        # swap can't corrupt the base. Accept commits + unlocks. ---
        rail = ui["rail"]
        back_btn, next_btn = ui["nav"]
        LOCK = (list(rail) + [back_btn, next_btn,
                swap["run_face"], swap["retry_face"], swap["accept_face"],
                swap["run_body"], swap["retry_body"], swap["accept_body"]])

        def _lockset(which):  # which: None (idle) / "face" / "body"
            idle = which is None
            ups = [gr.update(interactive=idle) for _ in rail]
            ups += [gr.update(interactive=idle), gr.update(interactive=idle)]  # back, next
            # The Run button doubles as Discard while its own swap is under review.
            if which == "face":
                ups += [gr.update(value="✕ Discard", variant="secondary", interactive=True),
                        gr.update(interactive=True), gr.update(interactive=True)]
            else:
                ups += [gr.update(value="Run face swap", variant="primary", interactive=idle),
                        gr.update(interactive=False), gr.update(interactive=False)]
            if which == "body":
                ups += [gr.update(value="✕ Discard", variant="secondary", interactive=True),
                        gr.update(interactive=True), gr.update(interactive=True)]
            else:
                ups += [gr.update(value="Run body swap", variant="primary", interactive=idle),
                        gr.update(interactive=False), gr.update(interactive=False)]
            return ups

        def _run_face(state, model, target_base, face_src, enhancer, strength, blend,
                      use_adet, adet_pos, adet_neg):
            if not target_base:
                raise gr.Error("Generate/select a base image first (step 3).")
            if not face_src:
                raise gr.Error("Provide a face source image.")
            self._require(["inswapper_128", "buffalo_l"], "Face swap")
            use_adet = bool(use_adet)
            if use_adet:
                backend, ident = discovery.parse_model_value(model)
                if backend != "sd":
                    raise gr.Error("ADetailer needs an SDXL/Pony/Illustrious model selected.")
                self._require(models.BODY_SWAP_KEYS, "ADetailer")
            gen_sd.release_sd()
            if not self.acquire_gpu(state):
                raise gr.Error("GPU is busy.")
            try:
                import time
                img = self._face_pipe().swap(
                    source_path=face_src, target_path=target_base,
                    enhancer=(enhancer or None),
                    blend_ratio=float(blend), enhancer_strength=float(strength))
                out = paths.cache_dir() / "swap"; out.mkdir(parents=True, exist_ok=True)
                p = out / f"face_{int(time.time())}.png"; img.save(p)
                res = str(p)
                if use_adet:
                    self._release_faceswap()
                    res = gen_sd.run_adetailer(ident, res, adet_pos, adet_neg,
                                               "DPM++ 2M", "Karras")
            finally:
                self.release_gpu(state)
            return [res] + _lockset("face")

        face_in = [self.state, s["model"], base["selected_base"], swap["face_source"],
                   swap["face_enhancer"], swap["face_enhancer_strength"], swap["face_blend_ratio"],
                   swap["face_adetailer"], swap["face_adet_pos"], swap["face_adet_neg"]]

        def _face_click(mode, *args):
            if mode == "review":  # Run button is showing Discard → drop the attempt
                return [None, "idle"] + _lockset(None)
            res = _run_face(*args)  # [result] + _lockset("face")
            return [res[0], "review"] + res[1:]

        swap["run_face"].click(_face_click, inputs=[swap["face_mode"]] + face_in,
                               outputs=[swap["result"], swap["face_mode"]] + LOCK)
        swap["retry_face"].click(_run_face, inputs=face_in, outputs=[swap["result"]] + LOCK)
        swap["accept_face"].click(lambda res: [res or gr.update(), "idle"] + _lockset(None),
                                  inputs=[swap["result"]],
                                  outputs=[base["selected_base"], swap["face_mode"]] + LOCK)

        def _run_body(state, model, sel_base, body_src, ip_scale, denoise, body_cfg,
                      cn_strength, steps, seed, sampler, scheduler, pos, neg, adet,
                      adet_pos, adet_neg, progress=gr.Progress()):
            if not sel_base:
                raise gr.Error("Generate/select a base image first (step 3).")
            if not body_src:
                raise gr.Error("Provide a body source image.")
            backend, ident = discovery.parse_model_value(model)
            if backend != "sd":
                raise gr.Error("Body swap needs an SDXL/Pony/Illustrious model selected.")
            self._require(models.BODY_SWAP_KEYS, "Body swap")
            self._release_faceswap()
            if not self.acquire_gpu(state):
                raise gr.Error("GPU is busy.")
            try:
                progress(0.1, desc="Segmenting + posing + body double…")
                out = gen_sd.body_swap(
                    ident, sel_base, body_src, pos, neg,
                    cn_strength=float(cn_strength), ip_scale=float(ip_scale),
                    denoise=float(denoise), cfg=float(body_cfg), steps=int(steps),
                    seed=int(seed), sampler=sampler, scheduler=scheduler,
                    adetailer=bool(adet), adet_prompt=adet_pos, adet_neg=adet_neg,
                    progress=progress)
            finally:
                self.release_gpu(state)
            if not out:
                if gen_sd.was_aborted():
                    gr.Info("Body swap aborted.")
                    return [None] + _lockset(None)
                raise gr.Error("Body swap produced no image.")
            return [out] + _lockset("body")

        body_in = [self.state, s["model"], base["selected_base"], swap["body_source"],
                   swap["body_ip_scale"], swap["body_denoise"], swap["body_cfg"],
                   swap["body_cn_strength"], s["steps"], s["seed"], s["sampler"],
                   s["scheduler"], base["pos"], base["neg"], swap["adetailer"],
                   swap["body_adet_pos"], swap["body_adet_neg"]]
        def _body_click(mode, *args, progress=gr.Progress()):
            if mode == "review":  # Run button is showing Discard → drop the attempt
                return [None, "idle"] + _lockset(None)
            res = _run_body(*args, progress=progress)  # [result] + _lockset(...)
            new_mode = "review" if res[0] else "idle"  # aborted/failed → idle
            return [res[0], new_mode] + res[1:]

        def _show_abort():   # reset + reveal the abort button as a swap starts
            return gr.update(visible=True, value="⛔ Abort body swap", interactive=True)

        def _hide_abort():
            return gr.update(visible=False)

        def _do_abort():
            gen_sd.request_abort()
            return gr.update(value="Aborting…", interactive=False)

        swap["run_body"].click(_show_abort, None, [swap["abort_body"]]).then(
            _body_click, inputs=[swap["body_mode"]] + body_in,
            outputs=[swap["result"], swap["body_mode"]] + LOCK).then(
            _hide_abort, None, [swap["abort_body"]])
        swap["retry_body"].click(_show_abort, None, [swap["abort_body"]]).then(
            _run_body, inputs=body_in, outputs=[swap["result"]] + LOCK).then(
            _hide_abort, None, [swap["abort_body"]])
        swap["abort_body"].click(_do_abort, None, [swap["abort_body"]])
        swap["accept_body"].click(lambda res: [res or gr.update(), "idle"] + _lockset(None),
                                  inputs=[swap["result"]],
                                  outputs=[base["selected_base"], swap["body_mode"]] + LOCK)

        # ADetailer prompt rows: face row shows only when Enhancer == adetailer;
        # body row follows its ADetailer checkbox.
        swap["face_adetailer"].change(
            lambda v: gr.update(visible=bool(v)),
            inputs=[swap["face_adetailer"]], outputs=[swap["face_adet_row"]])
        swap["adetailer"].change(
            lambda v: gr.update(visible=bool(v)),
            inputs=[swap["adetailer"]], outputs=[swap["body_adet_row"]])

        # A/B compare: base vs swapped, side by side, each full-screen + zoomable.
        def _ab(base_img, result_img):
            return (gr.update(visible=True), gr.update(visible=True),
                    base_img or None, result_img or None)
        swap["ab_btn"].click(_ab, inputs=[base["selected_base"], swap["result"]],
                             outputs=[swap["ab_row"], swap["ab_close_row"],
                                      swap["ab_base"], swap["ab_result"]])
        swap["ab_close"].click(
            lambda: (gr.update(visible=False), gr.update(visible=False)),
            outputs=[swap["ab_row"], swap["ab_close_row"]])

        # -- Touch Up: Inpaint mode (mask) + Cohesion mode (img2img normalize) --
        inp = c["inpaint"]
        inp["load_base"].click(lambda b: b, inputs=[base["selected_base"]],
                               outputs=[inp["editor"]])

        # Mode radio swaps the two sub-panels.
        inp["touchup_mode"].change(
            lambda m: (gr.update(visible=(m == "Inpaint")),
                       gr.update(visible=(m == "Cohesion"))),
            inputs=[inp["touchup_mode"]],
            outputs=[inp["inpaint_grp"], inp["cohesion_grp"]])

        # Keep both Touch Up sources mirroring the current base automatically.
        base["selected_base"].change(lambda b: (b or gr.update(), b or gr.update()),
                                     inputs=[base["selected_base"]],
                                     outputs=[inp["editor"], inp["cohesion_src"]])

        def _editor_mask(ev):
            import numpy as np
            from PIL import Image
            bg = ev.get("background") if isinstance(ev, dict) else None
            if bg is None:
                raise gr.Error("Load the base into the editor first.")
            H, W = bg.shape[:2]
            m = np.zeros((H, W), "uint8")
            for L in (ev.get("layers") or []):
                if getattr(L, "ndim", 0) == 3 and L.shape[2] == 4:
                    m = np.maximum(m, (L[..., 3] > 0).astype("uint8") * 255)
                elif getattr(L, "ndim", 0) == 3:
                    m = np.maximum(m, (L[..., :3].sum(2) > 0).astype("uint8") * 255)
            return bg, Image.fromarray(m, "L")

        _FILL = {"fill": 0, "original": 1, "latent noise": 2, "latent nothing": 3}

        def _run_inpaint(state, model, ev, ip_prompt, ip_neg, ip_denoise, count,
                         mask_blur, fill, full_res, padding, gallery,
                         steps, cfg, seed, sampler, scheduler, clip_skip,
                         progress=gr.Progress()):
            backend, ident = discovery.parse_model_value(model)
            if backend != "sd":
                raise gr.Error("Inpaint needs an SDXL/Pony/Illustrious model selected.")
            from PIL import Image
            bg, mask = _editor_mask(ev)
            if mask.getextrema() == (0, 0):
                raise gr.Error("Paint the area to fix first.")
            gen_sd.release_inpaint()
            self._release_faceswap()
            if not self.acquire_gpu(state):
                return gr.update()
            try:
                import os
                import tempfile
                tp = os.path.join(tempfile.mkdtemp(), "inp_src.png")
                Image.fromarray(bg[..., :3] if bg.ndim == 3 else bg).save(tp)
                outs = gen_sd.inpaint(ident, tp, mask, ip_prompt, ip_neg,
                                      denoise=float(ip_denoise), steps=int(steps),
                                      cfg=float(cfg), seed=int(seed), sampler=sampler,
                                      scheduler=scheduler, clip_skip=int(clip_skip),
                                      mask_blur=int(mask_blur),
                                      inpainting_fill=_FILL.get(fill, 1),
                                      full_res=bool(full_res), padding=int(padding),
                                      batch_size=int(count), progress=progress)
            finally:
                self.release_gpu(state)
            if not outs:
                return gr.update()
            history = [g[0] if isinstance(g, (list, tuple)) else g
                       for g in (gallery or [])]
            return history + list(outs)  # newest appended to the horizontal strip

        inp["run_inpaint"].click(
            _run_inpaint,
            inputs=[self.state, s["model"], inp["editor"], inp["inpaint_prompt"],
                    inp["inpaint_neg"], inp["inpaint_denoise"], inp["inpaint_count"],
                    inp["inpaint_mask_blur"], inp["inpaint_fill"], inp["inpaint_full_res"],
                    inp["inpaint_padding"], inp["inpaint_gallery"], s["steps"],
                    s["cfg_scale"], s["seed"], s["sampler"], s["scheduler"], s["clip_skip"]],
            outputs=[inp["inpaint_gallery"]])

        def _pick_inpaint(evt: gr.SelectData):
            return evt.value.get("image", {}).get("path") if isinstance(evt.value, dict) \
                else evt.value
        inp["inpaint_gallery"].select(_pick_inpaint, outputs=[inp["inpaint_picked"]])

        def _use_inpaint(picked, cur_base):
            if not picked:
                raise gr.Error("Click a result in the strip first.")
            return picked, cur_base  # new base, remember previous for Revert
        inp["use_inpaint"].click(_use_inpaint,
                                 inputs=[inp["inpaint_picked"], base["selected_base"]],
                                 outputs=[base["selected_base"], inp["inpaint_prev_base"]])
        inp["revert_inpaint"].click(lambda prev: prev or gr.update(),
                                    inputs=[inp["inpaint_prev_base"]],
                                    outputs=[base["selected_base"]])

        # Bounce a selected result between the two Touch Up modes as the new editable
        # image — does NOT assign it as base.
        def _send_to_cohesion(picked):
            if not picked:
                raise gr.Error("Click a result in the strip first.")
            return (picked, "Cohesion",
                    gr.update(visible=False), gr.update(visible=True))
        inp["send_to_cohesion"].click(
            _send_to_cohesion, inputs=[inp["inpaint_picked"]],
            outputs=[inp["cohesion_src"], inp["touchup_mode"],
                     inp["inpaint_grp"], inp["cohesion_grp"]])

        def _send_to_inpaint(picked):
            if not picked:
                raise gr.Error("Select a normalized result first.")
            return (picked, "Inpaint",
                    gr.update(visible=True), gr.update(visible=False))
        inp["send_to_inpaint"].click(
            _send_to_inpaint, inputs=[inp["cohesion_picked"]],
            outputs=[inp["editor"], inp["touchup_mode"],
                     inp["inpaint_grp"], inp["cohesion_grp"]])

        # Send a selected result back into the SAME subtab as the editable image.
        def _need(p):
            if not p:
                raise gr.Error("Select a result first.")
            return p
        inp["reuse_inpaint"].click(_need, inputs=[inp["inpaint_picked"]],
                                   outputs=[inp["editor"]])
        inp["reuse_cohesion"].click(_need, inputs=[inp["cohesion_picked"]],
                                    outputs=[inp["cohesion_src"]])

        # -- Cohesion mode: gentle img2img normalize using the character prompt --
        def _normalize(state, model, src, pos, neg, focus, cfg, steps,
                       progress=gr.Progress()):
            if not src:
                raise gr.Error("No base image to normalize (set a base on step ②).")
            backend, ident = discovery.parse_model_value(model)
            if backend != "sd":
                raise gr.Error("Cohesion needs an SDXL/Pony/Illustrious model selected.")
            from PIL import Image
            w, h = Image.open(src).size
            prompt = (pos or "").strip()
            if focus and focus.strip():
                prompt = (prompt + ", " + focus.strip()).strip(" ,")
            gen_sd.release_inpaint()
            self._release_faceswap()
            if not self.acquire_gpu(state):
                return gr.update()
            try:
                outs = gen_sd.generate_img2img(
                    ident, src, prompt, neg or "", int(w), int(h),
                    int(steps), float(cfg), -1, denoise=0.4, batch_size=4)
                return outs or gr.update()
            finally:
                self.release_gpu(state)

        inp["normalize_btn"].click(
            _normalize,
            inputs=[self.state, s["model"], inp["cohesion_src"], base["pos"], base["neg"],
                    inp["cohesion_focus"], inp["cohesion_cfg"], inp["cohesion_steps"]],
            outputs=[inp["cohesion_gallery"]])

        def _pick_cohesion(evt: gr.SelectData):
            return evt.value.get("image", {}).get("path") if isinstance(evt.value, dict) \
                else evt.value
        inp["cohesion_gallery"].select(_pick_cohesion, outputs=[inp["cohesion_picked"]])
        inp["use_cohesion"].click(lambda p: p or gr.update(),
                                  inputs=[inp["cohesion_picked"]],
                                  outputs=[base["selected_base"]])

        # -- step 6: pose variants (+ mandatory base-face swap) --
        def _gen_poses(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                       width, height, pos, neg, sel_base, progress=gr.Progress()):
            if not sel_base:
                raise gr.Error("Generate/select a base image first (step 3).")
            if not (pos and pos.strip()):
                raise gr.Error("Need a positive prompt (step 2).")
            self._require(["inswapper_128", "buffalo_l"], "Pose generation (base-face swap)")
            P = poses.POSES

            # Pass 1 — generate every pose with ONLY the generator resident
            # (face-swap models released first so the SD model has the whole GPU).
            self._release_faceswap()
            raw = []
            for i, ps in enumerate(P):
                progress((i, len(P)), desc=f"Generating pose {i + 1}/{len(P)} ({ps.distance}/{ps.angle})")
                pw, ph = (max(width, height), min(width, height)) if ps.orientation == "landscape" \
                    else (min(width, height), max(width, height))
                p_pos = f"{pos}, {ps.description}"
                p_neg = poses.pose_negative_for(ps.distance, neg)
                sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                imgs = self._gen_image(state, model, p_pos, p_neg, pw, ph, steps, cfg,
                                       sd, sampler, scheduler, clip_skip)
                raw.append((imgs[0] if imgs else None,
                            {"distance": ps.distance, "angle": ps.angle,
                             "orientation": ps.orientation}))

            # Pass 2 — free the generator, then apply the base face to each pose
            # (only the swap models resident now). Never both families at once.
            gen_sd.release_sd()
            fp = self._face_pipe()
            out = paths.cache_dir() / "poses"
            out.mkdir(parents=True, exist_ok=True)
            gallery, specs = [], []
            for i, (img, spec) in enumerate(raw):
                if not img:
                    continue
                final = img
                progress((i, len(raw)), desc=f"Applying base face {i + 1}/{len(raw)}")
                try:
                    if self.acquire_gpu(state):
                        try:
                            swapped = fp.swap(source_path=sel_base, target_path=img)
                            fp_path = out / f"pose_{i + 1:03d}.png"
                            swapped.save(fp_path)
                            final = str(fp_path)
                        finally:
                            self.release_gpu(state)
                except Exception:
                    traceback.print_exc()
                gallery.append(final)
                specs.append(spec)

            self._release_faceswap()  # leave the GPU clean for Save / Train
            if not gallery:
                raise gr.Error("No poses were generated.")
            return gallery, {"poses": gallery, "specs": specs}

        pose["generate"].click(
            _gen_poses,
            inputs=[self.state] + SET + [prm["positive_prompt"], prm["negative_prompt"],
                                         base["selected_base"]],
            outputs=[pose["pose_gallery"], ui["poses_state"]])

    def _wire_enhancer(self, ui):
        """Wire the Prompt step's Enhance buttons to Wan2GP's native enhancer."""
        prm = ui["components"]["setup"]
        if not all(hasattr(self, a) for a in
                   ("exec_prompt_enhancer_engine", "get_state_model_type", "get_model_def", "state")):
            return  # globals not injected (older host) — leave buttons inert

        def _enhance(state, text, progress=gr.Progress()):
            if not (text and text.strip()):
                raise gr.Error("Enter or seed a prompt first.")
            # The enhancer load / first-run model download happens inside Wan2GP and
            # doesn't stream to this bar — surface a status so it doesn't look frozen.
            progress(0.0, desc="Enhancing with Qwen3.5… (first run downloads the model — "
                               "watch the console)")
            model_type = self.get_state_model_type(state)
            model_def = self.get_model_def(model_type)
            out = self.exec_prompt_enhancer_engine(
                state, model_type, model_def,
                "T",            # text-only enhancement mode
                [text],         # original_prompts
                [None],         # image_start
                None,           # original_image_refs
                True,           # is_image
                False,          # audio_only
                -1,             # seed
                progress,
                -1,             # override_profile
                enhancer_kwargs={"image_prompt_type": "", "video_prompt_type": "",
                                 "audio_prompt_type": ""},
            )
            if out and out[0]:
                res = out[0]
                return res[0] if isinstance(res, (list, tuple)) else res
            return gr.update()

        prm["enhance_pos"].click(_enhance, inputs=[self.state, prm["positive_prompt"]],
                                 outputs=[prm["positive_prompt"]])
        prm["enhance_neg"].click(_enhance, inputs=[self.state, prm["negative_prompt"]],
                                 outputs=[prm["negative_prompt"]])


Plugin = ReplicantCharLab
