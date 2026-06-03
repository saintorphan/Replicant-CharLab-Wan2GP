"""Replicant Character Lab — a Wan2GP plugin.

A guided 7-step wizard (Info → Prompt → Base → Face/Body → Poses → Save → Train)
for building a reusable character and its LoRA datasets, ported from
SupremeDiffusion's PySide6 character creator into Wan2GP's Gradio UI.

NOTE: not an official plugin. Distribute via the plugin-manager "add from GitHub
URL" flow; do not add to the bundled plugins.json without dbm's approval.
"""
import logging
import random as _rng
import traceback

import gradio as gr

logger = logging.getLogger("replicant.plugin")

from shared.utils.plugins import WAN2GPPlugin

try:  # GPU arbitration with the main Video Generator + other plugins (e.g. ImageSuite)
    from shared.utils.process_locks import (acquire_GPU_ressources,
                                            any_GPU_process_running,
                                            release_GPU_ressources)
    _HAVE_LOCKS = True
except Exception:  # pragma: no cover
    _HAVE_LOCKS = False

from .core import (character, discovery, faceswap, gen_sd, models, paths, poses,
                   presets)
from .ui import wizard
from .ui.styles import CSS

PLUGIN_ID = "ReplicantCharLab"
PLUGIN_NAME = "Replicant Character Lab"

# Shared, idempotent right-click context menu ANYWHERE in the app + a cross-plugin
# presence handler so items can be wired regardless of plugin load order.
# First of the user's plugins to load creates window.SaintorphanMenu:
#   .register(match, label, handler)  match = 'image' | 'video' | any CSS selector
#       (e.g. '.reel2reel-timeline'); item shows only when the right-clicked element
#       matches. handler(matchedEl) runs on click.
#   .announce(name)                   a plugin declares itself present.
#   .whenPresent(name, cb)            cb fires now if present, else when name announces
#       — works whether the other plugin loaded earlier OR is installed/loaded later.
# Replicant announces 'replicant' and registers 'Replicant (Reference)' for images.
# Injected via <img onerror> (gr.HTML doesn't run <script>); all-single-quoted JS.
_CTX_MENU_JS = (
    "<img src=x style='display:none' onerror=\"(function(){"
    "if(!window.SaintorphanMenu){var M=window.SaintorphanMenu={items:[],present:{},_w:{}};"
    "M.announce=function(n){M.present[n]=true;(M._w[n]||[]).forEach(function(f){"
    "try{f();}catch(e){console.error(e);}});M._w[n]=[];};"
    "M.whenPresent=function(n,cb){if(M.present[n]){try{cb();}catch(e){console.error(e);}}"
    "else{(M._w[n]||(M._w[n]=[])).push(cb);}};"
    "M.register=function(match,label,handler){M.items.push("
    "{match:match,label:label,handler:handler});};"
    "M.srcOf=function(el){if(!el)return '';"
    "var a=el.getAttribute&&el.getAttribute('data-media-src');if(a)return a;"
    "if(el.currentSrc||el.src)return el.currentSrc||el.src;"
    "var q=el.querySelector&&el.querySelector('img,video');"
    "return q?(q.currentSrc||q.src||''):'';};"
    "function hit(match,el){if(match==='image')return el.closest('img');"
    "if(match==='video')return el.closest('video');"
    "try{return el.closest(match);}catch(e){return null;}}"
    "function close(){var m=document.getElementById('saintorphan-ctx');if(m)m.remove();}"
    "function build(x,y,hits){close();"
    "var menu=document.createElement('div');menu.id='saintorphan-ctx';"
    "menu.style.cssText='position:fixed;z-index:99999;background:#1f2430;border:1px solid "
    "#3a3f4b;border-radius:8px;padding:4px 0;box-shadow:0 6px 24px rgba(0,0,0,.5);"
    "min-width:210px;font-family:sans-serif;font-size:13px;color:#e5e7eb;';"
    "var h=document.createElement('div');h.textContent='OrphanSuite';"
    "h.style.cssText='padding:4px 14px;font-weight:700;color:#e83e8c;cursor:default;"
    "user-select:none;';menu.appendChild(h);"
    "var hr=document.createElement('div');hr.style.cssText='height:1px;background:#3a3f4b;"
    "margin:4px 0;';menu.appendChild(hr);"
    "hits.forEach(function(hk){var el=document.createElement('div');el.textContent=hk.it.label;"
    "el.style.cssText='padding:6px 14px;cursor:pointer;white-space:nowrap;';"
    "el.onmouseenter=function(){el.style.background='#2d3340';};"
    "el.onmouseleave=function(){el.style.background='';};"
    "el.addEventListener('click',function(ev){ev.stopPropagation();close();"
    "try{hk.it.handler(hk.el);}catch(err){console.error(err);}});menu.appendChild(el);});"
    "document.body.appendChild(menu);var r=menu.getBoundingClientRect();"
    "if(x+r.width>window.innerWidth)x=window.innerWidth-r.width-6;"
    "if(y+r.height>window.innerHeight)y=window.innerHeight-r.height-6;"
    "menu.style.left=x+'px';menu.style.top=y+'px';}"
    "document.addEventListener('contextmenu',function(e){var hits=[];"
    "M.items.forEach(function(it){var el=hit(it.match,e.target);if(el)hits.push({it:it,el:el});});"
    "if(!hits.length)return;e.preventDefault();build(e.clientX,e.clientY,hits);},true);"
    "document.addEventListener('click',close);document.addEventListener('scroll',close,true);}"
    "var M=window.SaintorphanMenu;if(!M._replicant){M._replicant=true;M.announce('replicant');"
    "var toRef=function(el){var src=M.srcOf(el);if(!src)return;"
    "var send=function(v){var b=document.querySelector('#replicant-ctx-relay textarea')"
    "||document.querySelector('#replicant-ctx-relay input');"
    "if(b){b.value=v+'|'+Date.now();b.dispatchEvent(new Event('input',{bubbles:true}));}};"
    "if(src.indexOf('/file=')>=0||src.indexOf('data:')===0){send(src);}"
    "else{fetch(src).then(function(r){return r.blob();}).then(function(bl){"
    "var fr=new FileReader();fr.onload=function(){send(fr.result);};fr.readAsDataURL(bl);})"
    ".catch(function(){send(src);});}};"
    "M.register('image','Replicant (Reference)',toRef);"
    "M.whenPresent('reel2reel',function(){"
    "M.register('.r2r-timeline-clip','Replicant (Reference)',toRef);});}"
    "})()\">")

# Draw a colored status badge in each pose thumbnail's corner from the state array
# (none=grey / approve=green ✓ / regen=red ✗). gr.Gallery has no per-item overlay API,
# so we paint it in the DOM; pointer-events:none keeps clicks reaching the thumbnail.


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
        self.request_global("get_lora_dir")           # per-native-model LoRA folder

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

    # -- GPU arbitration: shared with the base app + other plugins (ImageSuite) ---
    def _free_all_vram(self):
        """Release every model Replicant may have resident. Used as the GPU-resident
        reclaim callback so the base app / another plugin can take the GPU."""
        try:
            gen_sd.release_sd()
            gen_sd.release_inpaint()
            gen_sd.release_segmentation()  # cached BiRefNet (~1GB)
            self._release_faceswap()
            gen_sd._free_torch()
        except Exception:
            traceback.print_exc()

    def acquire_gpu(self, state):
        """Acquire the shared GPU lock — BLOCKS (raises) if ANY inference is running
        (base app or another plugin). Frees other registered residents on acquire."""
        if not _HAVE_LOCKS:
            return True
        if any_GPU_process_running(state, PLUGIN_ID):
            raise gr.Error("GPU is busy — another generation (the Video Generator or "
                           "another plugin) is running. Wait for it to finish.")
        acquire_GPU_ressources(state, PLUGIN_ID, PLUGIN_NAME, gr=gr)
        return True

    def release_gpu(self, state):
        """Release the lock but keep our pipelines cached, registering a reclaim
        callback so the base app / ImageSuite can free our VRAM when they need the GPU."""
        if not _HAVE_LOCKS:
            return
        try:
            release_GPU_ressources(state, PLUGIN_ID, keep_resident=True,
                                   process_name=PLUGIN_NAME,
                                   release_vram_callback=self._free_all_vram,
                                   force_release_on_acquire=True)
        except TypeError:  # older host without the resident kwargs
            release_GPU_ressources(state, PLUGIN_ID)

    @staticmethod
    def _require(keys, what):
        """Gate a GPU action on explicitly-downloaded models — never auto-pull."""
        miss = models.missing(keys)
        if miss:
            raise gr.Error(f"{what} needs models you haven't downloaded yet — get them "
                           f"in OrphanSuite → Models first: " + ", ".join(miss))

    # -- UI -----------------------------------------------------------------
    def _native_model_types(self):
        defs = getattr(self, "models_def", None) or {}
        try:
            return [mt for mt in defs if discovery.categorize_native(mt)]
        except Exception:
            return []

    def _loras_for_model(self, model_value):
        """Family-scoped LoRA choices [(label, value)] for a model dropdown value:
        native → the model's own Wan2GP LoRA dir; SD → name-based family filter."""
        backend, ident = discovery.parse_model_value(model_value or "")
        if backend == "native":
            return self._native_loras(ident)
        if backend == "sd":
            return discovery.lora_choices(family=discovery.model_family(model_value))
        return []

    def _reconcile_loras_to_model(self, ui):
        """On load the persisted LoRA selection is restored regardless of the
        restored model, but LoRAs are family-specific — so a selection saved with a
        different family/backend would show as ACTIVE for the wrong model (e.g. an
        SDXL LoRA active under a Z-Image model). Re-point the LoRA dropdowns to the
        restored model's family and drop any selected LoRA that isn't valid for it
        (valid ones are kept). _on_model handles this on user model changes; this
        does it for the initial render, which has no change event."""
        s = ui.get("settings") or {}
        model_dd = s.get("model")
        if model_dd is None:
            return
        mv = getattr(model_dd, "value", "") or ""
        choices = self._loras_for_model(mv)
        valid = {v for _, v in choices}
        inp = (ui.get("components") or {}).get("inpaint") or {}
        for dd in (s.get("loras"), inp.get("inpaint_loras")):
            if dd is None:
                continue
            dd.choices = list(choices)
            cur = dd.value if isinstance(dd.value, list) else ([dd.value] if dd.value else [])
            dd.value = [v for v in cur if v in valid]
        # Resolution tiers are family-specific too — repopulate for the restored
        # model and keep the persisted value only if it's a valid tier.
        res_dd = s.get("resolution")
        if res_dd is not None and mv:
            tiers = presets.resolution_tiers(mv, self.get_default_settings)
            res_dd.choices = list(tiers)
            valid_res = {v for _, v in tiers}
            if getattr(res_dd, "value", None) not in valid_res:
                res_dd.value = presets.recommended_resolution(mv, self.get_default_settings)

    def _native_loras(self, model_type):
        """LoRA files in this native model's own Wan2GP LoRA dir → dropdown
        (label, value). Each native family (Flux/Z-Image/Qwen) has its own dir, so
        this inherently shows only that family's LoRAs."""
        getter = getattr(self, "get_lora_dir", None)
        if not callable(getter):
            return []
        try:
            import os
            d = getter(model_type)
            if not (d and os.path.isdir(d)):
                return []
            return [(f, f) for f in sorted(os.listdir(d))
                    if f.lower().endswith((".safetensors", ".sft", ".lora", ".ckpt", ".pt"))]
        except Exception:
            return []

    @staticmethod
    def _native_lora_settings(selected, mult_str):
        """activated_loras (filenames, as the dropdown supplies) + loras_multipliers
        (space-separated) for native Wan2GP task settings. {} when none selected."""
        names = list(selected or [])
        if not names:
            return {}
        weights = [t.strip() for t in (mult_str or "").replace(";", ",").split(",")
                   if t.strip()]
        mult = " ".join(weights[:len(names)]) if weights else ""
        return {"activated_loras": names, "loras_multipliers": mult}

    @staticmethod
    def _sd_lora_list(selected, mult_str):
        """Selected SD LoRA paths + multiplier string → [{"name","weight"}] for the
        SD pipeline (gen_sd._apply_loras). Empty list when none selected. The last
        given multiplier fills any unspecified trailing LoRAs."""
        sel = list(selected or [])
        if not sel:
            return []
        try:
            ws = [float(x) for x in (mult_str or "").replace(";", ",").split(",")
                  if x.strip()]
        except ValueError:
            ws = []
        return [{"name": p, "weight": (ws[i] if i < len(ws)
                                       else (ws[-1] if ws else 1.0))}
                for i, p in enumerate(sel)]

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
        # NOTE: deliberately NO MutationObserver — a permanent document-wide observer
        # interfered with other plugins' re-renders (the plugin-manager list flickered
        # away). Just re-apply the class a few times after the SPA settles.
        gr.HTML(
            "<img src=x style='display:none' onerror=\"(function(){"
            "var NAME=" + repr(PLUGIN_NAME) + ";"
            "function mark(){document.querySelectorAll("
            "'.tab-nav button,button[role=&quot;tab&quot;]').forEach(function(b){"
            "if(b.textContent.trim()===NAME)b.classList.add('replicant-tabbtn');});}"
            "[200,800,2000,4000].forEach(function(t){setTimeout(mark,t);});})()\">",
            elem_classes="replicant-hidden")
        # Shared right-click context menu (app-wide) + Replicant's item.
        gr.HTML(_CTX_MENU_JS, elem_classes="replicant-hidden")
        with gr.Column(elem_id="replicant-root"):
            ui = wizard.build_wizard(model_choices=model_choices, lora_choices=lora_choices,
                                     init=init)
        # Guard the post-build wiring: a single bad event wiring should degrade this
        # tab, not abort the whole Wan2GP Blocks build (which would take down the app).
        for _step in (self._reconcile_loras_to_model, self._wire_enhancer,
                      self._wire_generation, self._wire_context_menu):
            try:
                _step(ui)
            except Exception:
                traceback.print_exc()
                gr.Warning(f"Replicant: a UI wiring step failed ({_step.__name__}); "
                           "that feature may be inert.")
        self.on_tab_outputs = [self.main_tabs] if hasattr(self, "main_tabs") else None
        self._ui = ui
        return ui

    # -- generation backends ------------------------------------------------
    @staticmethod
    def _res(resolution, orientation="portrait"):
        """(w, h) for the locked 'WxH' portrait resolution in a pose orientation:
        portrait (as-is), square 1:1 (sitting), or landscape (reclining/laying)."""
        return presets.oriented(resolution, orientation)

    def _is_square_native(self, model) -> bool:
        """True for 1:1-trained native models (Z-Image, Flux 2 Klein). They're
        trained at 1024² square and paint a vertically-stretched 'big head' when
        forced to portrait — so we render them square instead. Gated on a square
        (w==h) default resolution too, which leaves the landscape Z-Image *Control*
        variants (1920x1088) on their own native aspect."""
        cache = self.__dict__.setdefault("_sqnat_cache", {})
        if model in cache:
            return cache[model]
        try:
            backend, ident = discovery.parse_model_value(model)
        except Exception:
            return False
        result = False
        if backend == "native" and ident and any(
                f in ident for f in ("z_image", "flux2_klein")):
            try:
                res = str(self.get_default_settings(ident).get("resolution", ""))
                w, h = (int(x) for x in res.lower().split("x"))
                result = (w == h)
            except Exception:
                result = True  # these families default square — assume square
        cache[model] = result
        return result

    def _orient_for(self, model, orientation):
        """Force 'square' for 1:1-native models; otherwise honour the pose's
        orientation (portrait/square/landscape). Applied at every gen entry point
        so a Z-Image / Flux-2-Klein base, reimagine, pose and re-roll all stay on
        the model's native square instead of a distorting portrait."""
        return "square" if self._is_square_native(model) else orientation

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
                   sampler, scheduler, clip_skip, loras=None, lora_mult="", api=None):
        """One image via the routed backend (txt2img). Returns saved paths.
        ``loras`` are the family-scoped selections from the settings bar. ``api``
        is forwarded to the native path (see _gen_native re: webui-wrapping)."""
        backend, ident = discovery.parse_model_value(model_value)
        if backend == "native":
            return self._gen_native(ident, pos, neg, w, h, steps, cfg, seed,
                                    loras=loras, mult_str=lora_mult, api=api)
        if backend == "sd":
            if not self.acquire_gpu(state):
                return []
            try:
                return gen_sd.generate_txt2img(
                    ident, pos, neg, w, h, steps, cfg, seed,
                    sampler=sampler or "DPM++ 2M", scheduler=scheduler or "",
                    clip_skip=int(clip_skip),
                    loras=self._sd_lora_list(loras, lora_mult))
            finally:
                self.release_gpu(state)
        raise gr.Error("Select a model from the dropdown first.")

    def _native_caps(self, model_type) -> dict:
        """The native model's capability/def dict (inpaint_support = has a
        guide-image input, …). Prefer get_model_def; fall back to models_def."""
        getter = getattr(self, "get_model_def", None)
        if callable(getter):
            try:
                d = getter(model_type)
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        defs = getattr(self, "models_def", None) or {}
        d = defs.get(model_type)
        return d if isinstance(d, dict) else {}

    def _gen_native(self, model_type, pos, neg, w, h, steps, cfg, seed, *,
                    mode="txt2img", denoise=1.0, guide_path=None,
                    loras=None, mult_str="", api=None):
        """One native (Flux/Z-Image/Qwen) image via Wan2GP's own task queue.

        mode: 'txt2img' (image_mode 1) or 'img2img' (image_mode 1 + image_guide +
        denoise). img2img needs a model with a guide-image input (inpaint_support)
        — Flux / Qwen-Image / Z-Image *Control*; plain z_image_base has none.
        ``loras`` are this model's own LoRA filenames (+ ``mult_str`` multipliers).
        Native gens run in Wan2GP's queue — they do NOT take the plugin's SD GPU
        lock, so callers must not wrap this in acquire_gpu/release_gpu.

        ``api`` MUST be the session the click handler closed over: Wan2GP only
        drives a submitted task when it detects the api session in the handler's
        closure (see api_webui._callback_uses_api_session) — otherwise the task is
        admitted to the queue but never runs (silent hang)."""
        if int(seed) < 0:
            seed = _rng.randint(0, 2**31 - 1)
        settings = dict(self.get_default_settings(model_type))
        settings.update({
            "model_type": model_type, "image_mode": 1, "prompt": pos,
            "negative_prompt": neg or "", "resolution": f"{int(w)}x{int(h)}",
            "num_inference_steps": int(steps), "guidance_scale": float(cfg),
            "seed": int(seed), "video_length": 1, "batch_size": 1,
        })
        settings.update(self._native_lora_settings(loras, mult_str))
        if mode == "img2img":
            if not self._native_caps(model_type).get("inpaint_support"):
                raise gr.Error(
                    f"'{model_type}' has no guide-image input, so it can't img2img. "
                    "Use a Flux/Qwen-Image model, a Z-Image *Control* model, or an "
                    "SDXL/Pony/Illustrious checkpoint.")
            # Fit the guide to the exact target WITHOUT stretching first — Wan2GP
            # would otherwise resize-stretch a mismatched-aspect guide (e.g. a
            # portrait fed into a square Z-Image), squashing the subject into a
            # 'big head' that compounds over a chain of img2img hops and then
            # confuses the downstream face swap. See gen_sd.fit_image.
            guide_fit = gen_sd.fit_image(guide_path, w, h) if guide_path else guide_path
            # "VG": V = use the guide image, G = honour denoising_strength (partial
            # denoise). Without "G", wgp forces denoise=1.0 (full regen, ignoring
            # the slider); no "A" so no mask is required → a true img2img.
            settings.update({"image_guide": guide_fit,
                             "denoising_strength": float(denoise),
                             "video_prompt_type": "VG"})
        try:
            result = (api or self._api).submit_task(settings).result()
        except Exception as e:
            # An abort/cancel must NOT escape (it would blow up the whole pose batch
            # and lose every completed pose) — swallow it and let the caller keep
            # what's done. Genuine failures still raise.
            if gen_sd.was_aborted():
                return []
            if "generation in progress" in str(e).lower():
                raise gr.Error(
                    "Another generation is still pending. Native (Flux/Z-Image/Qwen) "
                    "gens run in Wan2GP's queue and PAUSE if the browser loses focus — "
                    "click the Video Generator tab to let it finish, then try again.")
            raise
        if gen_sd.was_aborted():  # cancelled mid-task → discard, keep prior poses
            return []
        if result.success and result.generated_files:
            return list(result.generated_files)
        if result.errors:
            if gen_sd.was_aborted():
                return []
            raise gr.Error(str(list(result.errors)[0]))
        return []

    def _pose_swaps(self, state, ident, items, pos, neg, do_body, body_src, body_ip,
                    body_den, apply_face, face_src, out, progress, start_idx=0, adet=None):
        """Order per pose: body double → body ADetailer (person) → face swap → face
        ADetailer (face). Body-before-face so the swapped face survives; ADetailer for
        each region uses its own model. ``adet`` = the Replicate tab's ADetailer dict.
        Returns (gallery, specs)."""
        from pathlib import Path
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        ad = adet or {}

        def _adet_pass(seq, detector, ppos, pneg, label):
            res = []
            for i, (img, spec) in enumerate(seq):
                final = img
                if img is not None:
                    progress((i, len(seq)), desc=f"{label} {i + 1}/{len(seq)}")
                    if self.acquire_gpu(state):
                        try:
                            r = gen_sd.run_adetailer(ident, img, ppos or pos, pneg or neg,
                                                     detector=detector)
                            final = r or img
                        except Exception:
                            traceback.print_exc()
                        finally:
                            self.release_gpu(state)
                res.append((final, spec))
            return res

        body_ok = body_fail = 0
        if do_body:
            gen_sd.release_sd(); self._release_faceswap()
            bodied = []
            for i, (img, spec) in enumerate(items):
                final = img
                if img is None:
                    bodied.append((None, spec)); continue
                progress((i, len(items)), desc=f"Body double {i + 1}/{len(items)}")
                try:
                    if self.acquire_gpu(state):
                        try:
                            r = gen_sd.body_swap(ident, img, body_src, pos, neg,
                                                 ip_scale=float(body_ip),
                                                 denoise=float(body_den),
                                                 adetailer=False, progress=progress)
                            final = r or img
                            if r:
                                body_ok += 1
                            else:  # None = aborted/segfail → original kept, no change
                                body_fail += 1
                                logger.warning("Body double pose %d: no result, kept "
                                               "original (aborted or seg failed).", i + 1)
                        finally:
                            self.release_gpu(state)
                except Exception:
                    body_fail += 1
                    logger.warning("Body double pose %d FAILED; kept original.", i + 1)
                    traceback.print_exc()
                bodied.append((final, spec))
            items = bodied
        # Body ADetailer (person model) — before the face swap.
        if ad.get("body"):
            gen_sd.release_sd(); self._release_faceswap()
            items = _adet_pass(items, "person", ad.get("body_pos"), ad.get("body_neg"),
                               "Body ADetailer")
        gen_sd.release_sd()
        gallery, specs = [], []
        face_ok = face_noface = face_fail = 0
        fp = self._face_pipe() if apply_face else None
        for i, (img, spec) in enumerate(items):
            final = img
            if apply_face and img is not None:
                progress((i, len(items)), desc=f"Applying face {i + 1}/{len(items)}")
                try:
                    if self.acquire_gpu(state):
                        try:
                            swapped = fp.swap(source_path=face_src, target_path=img)
                            fp_path = out / f"pose_{start_idx + i + 1:03d}.png"
                            swapped.save(fp_path); final = str(fp_path)
                            face_ok += 1
                        finally:
                            self.release_gpu(state)
                except ValueError as e:  # "No face detected in target/source image"
                    if "No face detected" in str(e):
                        face_noface += 1
                        logger.warning("Face swap pose %d skipped: %s", i + 1, e)
                    else:
                        face_fail += 1
                        traceback.print_exc()
                except Exception:
                    face_fail += 1
                    traceback.print_exc()
            gallery.append(final); specs.append(spec)
        self._release_faceswap()
        # Summary so both passes are observable (also in the terminal log).
        n = sum(1 for im, _ in items if im is not None)
        parts = []
        if do_body:
            parts.append(f"body double {body_ok}/{n}"
                         + (f" ({body_fail} kept original)" if body_fail else ""))
        if apply_face:
            parts.append(f"face swap {face_ok}/{n}"
                         + (f" ({face_noface} no-face skips)" if face_noface else "")
                         + (f" ({face_fail} errors)" if face_fail else ""))
        if parts:
            msg = "Replicate passes — " + "; ".join(parts) + "."
            logger.info(msg)
            try:
                gr.Info(msg)
            except Exception:
                pass
        # Face ADetailer (face model) — after the face swap.
        if ad.get("face"):
            gen_sd.release_sd()
            gallery = [g for g, _ in
                       _adet_pass(list(zip(gallery, specs)), "face",
                                  ad.get("face_pos"), ad.get("face_neg"), "Face ADetailer")]
            self._release_faceswap()
        gen_sd.release_sd()  # leave the GPU clean for Save / Train
        return gallery, specs

    def _persist_poses(self, gallery, specs):
        """Copy poses to the stable persist dir + save list/specs to wizard state so
        they survive an app reload. Returns the persisted paths (None preserved)."""
        import os
        import shutil
        from .core import wizard_state
        d = paths.cache_dir() / "persist" / "poses"
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
        out = []
        for i, p in enumerate(gallery):
            if p and isinstance(p, str) and os.path.isfile(p):
                dst = d / f"pose_{i:02d}.png"
                try:
                    shutil.copy2(p, dst); out.append(str(dst))
                except Exception:
                    out.append(p)
            else:
                out.append(None)
        wizard_state.update({"poses.pose_gallery": out, "poses.specs": specs})
        return out

    def _snapshot_prev(self, images):
        """Copy the current pose images to a SEPARATE 'prev' dir before a re-run
        overwrites the canonical ones, so the per-pose ↩ undo can restore them.
        Returns the snapshot paths (None preserved)."""
        import os
        import shutil
        d = paths.cache_dir() / "persist" / "poses_prev"
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
        out = []
        for i, p in enumerate(images or []):
            if p and isinstance(p, str) and os.path.isfile(p):
                dst = d / f"prev_{i:02d}.png"
                try:
                    shutil.copy2(p, dst); out.append(str(dst))
                except Exception:
                    out.append(None)
            else:
                out.append(None)
        return out

    def _wire_generation(self, ui):
        c, s = ui["components"], ui["settings"]
        base, prm, swap, pose = c["base"], c["setup"], c["swap"], c["poses"]
        if not getattr(self, "_api", None):
            return

        # Wan2GP only DRIVES a submitted native task when it sees the api session in
        # the click handler's closure (api_webui._callback_uses_api_session); without
        # it the task is admitted to the queue but never runs. So every handler that
        # can trigger a native gen closes over ``api`` and forwards it to _gen_native.
        api = self._api

        # SET is the shared settings-bar inputs passed to every generation handler.
        # 'resolution' is the portrait base 'WxH' (poses auto-orient from it); loras
        # + multipliers ride at the end so they reach base/pose/reimagine gen.
        SET = [s["model"], s["sampler"], s["scheduler"], s["steps"], s["cfg_scale"],
               s["clip_skip"], s["seed"], s["resolution"],
               s["loras"], s["lora_multipliers"]]

        # On model switch: scope LoRAs to the model's family AND auto-populate the
        # recommended cfg/steps/sampler/scheduler + the resolution tier list.
        #   - native (Flux/Z-Image/Qwen): LoRAs from that model's own Wan2GP dir.
        #   - SD (SDXL/Pony/Illustrious): LoRAs filtered by name-based family.
        # setting_keys maps the preset dict onto the settings-bar controls.
        setting_keys = ["sampler", "scheduler", "steps", "cfg", "clip_skip"]
        _key_to_ctl = {"cfg": "cfg_scale"}  # preset key → settings-bar component key

        def _on_model(model_value):
            backend, ident = discovery.parse_model_value(model_value)
            if backend == "native":
                cat = discovery.categorize_native(ident) or "Native"
                lora_up = gr.update(choices=self._native_loras(ident), value=[],
                                    label=f"{cat} LoRAs")
            elif backend == "sd":
                fam = discovery.model_family(model_value)
                lora_up = gr.update(choices=discovery.lora_choices(family=fam),
                                    value=[], label=f"{fam} LoRAs" if fam else "LoRAs")
            else:
                lora_up = gr.update(choices=[], value=[], label="LoRAs")
            rec = presets.for_model(model_value, self.get_default_settings)
            ups = [gr.update(value=rec[k]) if k in rec else gr.update()
                   for k in setting_keys]
            # Resolution: repopulate the tier list + select the recommended one.
            res_up = gr.update(
                choices=presets.resolution_tiers(model_value, self.get_default_settings),
                value=presets.recommended_resolution(model_value, self.get_default_settings))
            return [lora_up, lora_up] + ups + [res_up]

        s["model"].change(
            _on_model, inputs=[s["model"]],
            outputs=[s["loras"], c["inpaint"]["inpaint_loras"]]
            + [s[_key_to_ctl.get(k, k)] for k in setting_keys] + [s["resolution"]])

        # -- base candidates --
        def _gen_base(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                      resolution, loras, lora_mult, pos, neg, count,
                      progress=gr.Progress()):
            if not (pos and pos.strip()):
                raise gr.Error("Build or enhance a positive prompt on step 2 first.")
            gen_sd.clear_abort()  # don't inherit a stale abort flag from a prior run
            self._release_faceswap()  # base gen is pure SD — free InsightFace VRAM
            bw, bh = self._res(resolution, self._orient_for(model, "portrait"))  # base ref
            files, n = [], int(count)
            for i in range(n):
                sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                progress((i, n), desc=f"Base {i + 1}/{n}")
                files += self._gen_image(state, model, pos, neg, bw, bh,
                                         steps, cfg, sd, sampler, scheduler, clip_skip,
                                         loras=loras, lora_mult=lora_mult, api=api)
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

        # Reimagine the reference (img2img). SD-family runs locally; native
        # (Flux/Qwen/Z-Image) routes through Wan2GP's queue if it supports a guide.
        def _reimagine(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                       resolution, loras, lora_mult, pos, neg, denoise, count,
                       ref_img, progress=gr.Progress()):
            backend, ident = discovery.parse_model_value(model)
            if not backend:
                raise gr.Error("Select a model from the dropdown first.")
            if not ref_img:
                raise gr.Error("No reference image to reimagine (add one on step 1).")
            if not (pos and pos.strip()):
                raise gr.Error("Build or enhance a positive prompt on step 2 first.")
            gen_sd.clear_abort()  # don't inherit a stale abort flag from a prior run
            self._release_faceswap()
            rw, rh = self._res(resolution, self._orient_for(model, "portrait"))  # reimagine
            files, n = [], int(count)
            if backend == "native":
                for i in range(n):
                    sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                    progress((i, n), desc=f"Reimagining {i + 1}/{n} (img2img)")
                    files += self._gen_native(
                        ident, pos, neg, rw, rh, steps, cfg, sd,
                        mode="img2img", denoise=float(denoise), guide_path=ref_img,
                        loras=loras, mult_str=lora_mult, api=api)
            else:  # sd
                sd_loras = self._sd_lora_list(loras, lora_mult)
                if not self.acquire_gpu(state):
                    return gr.update(), gr.update()
                try:
                    for i in range(n):
                        sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                        progress((i, n), desc=f"Reimagining {i + 1}/{n} (img2img)")
                        files += gen_sd.generate_img2img(
                            ident, ref_img, pos, neg, rw, rh, steps, cfg, sd,
                            denoise=float(denoise), sampler=sampler, scheduler=scheduler,
                            clip_skip=int(clip_skip), loras=sd_loras)
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

        # Inpaint/Cohesion are native sub-tabs now (click either freely).
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

        def _parse_loras(sel, mult):
            if not sel:
                return []
            try:
                ws = [float(x) for x in (mult or "").split(",") if x.strip()]
            except ValueError:
                ws = []
            out = []
            for i, p in enumerate(sel):
                w = ws[i] if i < len(ws) else (ws[-1] if ws else 1.0)
                out.append({"name": p, "weight": w})
            return out

        def _run_inpaint(state, model, ev, ip_prompt, ip_neg, ip_denoise, count,
                         mask_blur, fill, full_res, padding, loras_sel, lora_mult, gallery,
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
                                      batch_size=int(count),
                                      loras=_parse_loras(loras_sel, lora_mult),
                                      progress=progress)
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
                    inp["inpaint_padding"], inp["inpaint_loras"], inp["inpaint_lora_mult"],
                    inp["inpaint_gallery"], s["steps"], s["cfg_scale"], s["seed"],
                    s["sampler"], s["scheduler"], s["clip_skip"]],
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

        # Bounce a selected result between the two Touch Up sub-tabs as the new
        # editable image (switches tabs) — does NOT assign it as base.
        def _send_to_cohesion(picked):
            if not picked:
                raise gr.Error("Click a result in the strip first.")
            return picked, gr.Tabs(selected="cohesion")
        inp["send_to_cohesion"].click(
            _send_to_cohesion, inputs=[inp["inpaint_picked"]],
            outputs=[inp["cohesion_src"], inp["touchup_tabs"]])

        def _send_to_inpaint(picked):
            if not picked:
                raise gr.Error("Select a normalized result first.")
            return picked, gr.Tabs(selected="inpaint")
        inp["send_to_inpaint"].click(
            _send_to_inpaint, inputs=[inp["cohesion_picked"]],
            outputs=[inp["editor"], inp["touchup_tabs"]])

        # Send a selected result back into the SAME subtab as the editable image.
        def _need(p):
            if not p:
                raise gr.Error("Select a result first.")
            return p
        inp["reuse_inpaint"].click(_need, inputs=[inp["inpaint_picked"]],
                                   outputs=[inp["editor"]])
        inp["reuse_cohesion"].click(_need, inputs=[inp["cohesion_picked"]],
                                    outputs=[inp["cohesion_src"]])

        # -- Cohesion mode: gentle img2img normalize (own prompts, this subtab only) --
        def _normalize(state, model, src, pos, neg, focus, cfg, steps,
                       loras, lora_mult, progress=gr.Progress()):
            if not src:
                raise gr.Error("No base image to normalize (set a base on step ②).")
            backend, ident = discovery.parse_model_value(model)
            if not backend:
                raise gr.Error("Select a model from the dropdown first.")
            from PIL import Image
            w, h = Image.open(src).size
            prompt = (pos or "").strip()
            if focus and focus.strip():
                prompt = (prompt + ", " + focus.strip()).strip(" ,")
            gen_sd.clear_abort()  # don't inherit a stale abort flag from a prior run
            self._release_faceswap()
            if backend == "native":  # img2img via Wan2GP's queue (no SD GPU lock)
                outs = self._gen_native(ident, prompt, neg or "", int(w), int(h),
                                        int(steps), float(cfg), -1, mode="img2img",
                                        denoise=0.4, guide_path=src,
                                        loras=loras, mult_str=lora_mult, api=api)
                return outs or gr.update()
            gen_sd.release_inpaint()
            if not self.acquire_gpu(state):
                return gr.update()
            try:
                outs = gen_sd.generate_img2img(
                    ident, src, prompt, neg or "", int(w), int(h),
                    int(steps), float(cfg), -1, denoise=0.4, batch_size=4,
                    loras=self._sd_lora_list(loras, lora_mult))
                return outs or gr.update()
            finally:
                self.release_gpu(state)

        inp["normalize_btn"].click(
            _normalize,
            inputs=[self.state, s["model"], inp["cohesion_src"], inp["cohesion_prompt"],
                    inp["cohesion_neg"], inp["cohesion_focus"], inp["cohesion_cfg"],
                    inp["cohesion_steps"], s["loras"], s["lora_multipliers"]],
            outputs=[inp["cohesion_gallery"]])

        def _pick_cohesion(evt: gr.SelectData):
            return evt.value.get("image", {}).get("path") if isinstance(evt.value, dict) \
                else evt.value
        inp["cohesion_gallery"].select(_pick_cohesion, outputs=[inp["cohesion_picked"]])
        inp["use_cohesion"].click(lambda p: p or gr.update(),
                                  inputs=[inp["cohesion_picked"]],
                                  outputs=[base["selected_base"]])

        # -- step 6: pose variants (+ mandatory base-face swap) --
        def _pose_adet(fa, fpos, fneg, ba, bpos, bneg, sdxl_ident):
            """Build the ADetailer dict for poses + require its models. Both ADetailer
            passes are SDXL inpaint, so they need an SDXL checkpoint (the main model
            if it's SD, else the Body-swap/detail model) — gate on sdxl_ident."""
            has_sdxl = bool(sdxl_ident)
            ad = {"face": bool(fa) and has_sdxl, "face_pos": fpos, "face_neg": fneg,
                  "body": bool(ba) and has_sdxl, "body_pos": bpos, "body_neg": bneg}
            if (bool(fa) or bool(ba)) and not has_sdxl:
                gr.Warning("Pose ADetailer needs an SDXL checkpoint — pick one in "
                           "'Body-swap / detail model' — skipping the detail pass.")
            if ad["face"]:
                self._require(["buffalo_l"], "Pose face ADetailer")
            if ad["body"]:
                self._require(["person_yolov8s_seg"], "Pose body ADetailer")
            return ad

        def _gen_poses(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                       resolution, loras, lora_mult, pos, neg, sel_base,
                       face_mode, body_mode, face_ref, body_ref, body_ip_scale,
                       body_denoise, pose_face_adet, pfa_pos, pfa_neg,
                       pose_body_adet, pba_pos, pba_neg, pose_sdxl,
                       progress=gr.Progress()):
            if not sel_base:
                raise gr.Error("Generate/select a base image first (step 3).")
            if not (pos and pos.strip()):
                raise gr.Error("Need a positive prompt (step 2).")
            backend, ident = discovery.parse_model_value(model)
            # SDXL checkpoint for the SDXL-only enhancement passes (body swap +
            # ADetailer): the main model when it's SD, else the chosen detail model —
            # so a native (Flux/Z-Image) pose can still get a body double.
            sdxl_ident = ident if backend == "sd" else (pose_sdxl or "")
            # Resolve the face/body sources from the None/Use Base/Use Reference modes.
            face_src = sel_base if face_mode == "Use Base" else (
                face_ref if face_mode == "Use Reference" else None)
            body_src = sel_base if body_mode == "Use Base" else (
                body_ref if body_mode == "Use Reference" else None)
            apply_face = bool(face_src)
            do_body = bool(body_src) and bool(sdxl_ident)
            if bool(body_src) and not sdxl_ident:
                gr.Warning("Body double needs an SDXL checkpoint — pick one in "
                           "'Body-swap / detail model' (a native pose model can't "
                           "body-swap itself) — skipping body double.")
            if do_body:
                self._require(models.BODY_SWAP_KEYS, "Body double for poses")
            if apply_face:
                self._require(["inswapper_128", "buffalo_l"], "Pose face swap")
            adet = _pose_adet(pose_face_adet, pfa_pos, pfa_neg,
                              pose_body_adet, pba_pos, pba_neg, sdxl_ident)
            P = poses.POSES
            # Replicate OVERRIDES the base prompt's framing: the base prompt carries
            # "standing front, full body, head to toe" to keep the BASE image a clean
            # reference, but here each pose's own description supplies the framing —
            # so strip the base framing or every pose snaps back to standing.
            base_clean = character.strip_base_framing(pos)

            # Pass 1 — generate every pose with ONLY the generator resident
            # (face-swap models released first so the SD model has the whole GPU).
            self._release_faceswap()
            gen_sd.clear_abort()  # fresh abort state for this batch
            raw = []
            display = [None] * len(P)  # streamed to the gallery as each pose lands
            for i, ps in enumerate(P):
                spec = {"distance": ps.distance, "angle": ps.angle,
                        "orientation": ps.orientation}
                if gen_sd.was_aborted():  # master abort → stop the whole batch
                    raw.append((None, spec))
                    continue
                if gen_sd.should_skip(i):  # this pose individually aborted → skip
                    raw.append((None, spec))
                    continue
                progress((i, len(P)), desc=f"Generating pose {i + 1}/{len(P)} ({ps.distance}/{ps.angle})")
                pw, ph = self._res(resolution, self._orient_for(model, ps.orientation))
                p_pos = ", ".join(p for p in (base_clean, ps.description) if p)
                p_neg = poses.pose_negative_for(ps.distance, neg)
                sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                gen_sd.set_current_index(i)
                imgs = self._gen_image(state, model, p_pos, p_neg, pw, ph, steps, cfg,
                                       sd, sampler, scheduler, clip_skip,
                                       loras=loras, lora_mult=lora_mult, api=api)
                # Interrupted mid-gen (master or this pose) → discard the partial.
                img = None if gen_sd.should_skip(i) else (imgs[0] if imgs else None)
                raw.append((img, spec))
                display[i] = img
                yield list(display)  # incremental: show each raw pose as it finishes

            out = paths.cache_dir() / "poses"
            # On a master abort, skip the (slow) swap pass and keep the raw gens.
            if gen_sd.was_aborted():
                gallery = [im for im, _ in raw]
                specs = [sp for _, sp in raw]
            else:
                # Keep all N slots aligned (None for any that failed to generate).
                # sdxl_ident (not the pose-gen ident) drives the SDXL swap/detail passes.
                gallery, specs = self._pose_swaps(
                    state, sdxl_ident, raw, pos, neg, do_body, body_src, body_ip_scale,
                    body_denoise, apply_face, face_src, out, progress, adet=adet)
            if not any(gallery):
                raise gr.Error("Aborted — no poses generated." if gen_sd.was_aborted()
                               else "No poses were generated.")
            saved = self._persist_poses(gallery, specs)
            return saved + [{"poses": saved, "specs": specs}]

        pose_in = [self.state] + SET + [prm["positive_prompt"], prm["negative_prompt"],
                                        base["selected_base"], pose["face_mode"],
                                        pose["body_mode"], swap["face_source"],
                                        swap["body_source"], swap["body_ip_scale"],
                                        swap["body_denoise"], pose["pose_face_adet"],
                                        pose["pose_face_adet_pos"], pose["pose_face_adet_neg"],
                                        pose["pose_body_adet"], pose["pose_body_adet_pos"],
                                        pose["pose_body_adet_neg"], pose["pose_sdxl_model"]]
        pose_out = pose["pose_imgs"] + [ui["poses_state"]]

        # -- Replicate aborts: master (stop the batch) + per-pose (skip/interrupt one) --
        def _abort_all():
            gen_sd.request_abort()
            try:
                self._api.cancel()  # interrupt a running native pose (Wan2GP queue)
            except Exception:
                pass
            gr.Info("Aborting — finishing/cancelling the current pose…")

        pose["abort_all"].click(_abort_all, None, None)

        def _make_pose_abort(i):
            def _h():
                gen_sd.request_skip(i)
                if gen_sd.current_index() == i:  # this pose is in-flight → cancel native too
                    try:
                        self._api.cancel()
                    except Exception:
                        pass
            return _h
        for _i, _btn in enumerate(pose["pose_abort"]):
            _btn.click(_make_pose_abort(_i), None, None)

        # Grey out ALL FOUR gen buttons while any run is in flight, re-enable after.
        # .then() (not .success()) runs even if the handler errored, so the buttons
        # never get stuck disabled.
        gen_btns = [pose["generate"], pose["rerun"],
                    pose["repass_face"], pose["repass_body"]]

        def _lock_btns():
            return [gr.update(interactive=False)] * 4

        def _unlock_btns():
            return [gr.update(interactive=True)] * 4

        pose["generate"].click(_lock_btns, None, gen_btns).then(
            _gen_poses, inputs=pose_in, outputs=pose_out).then(
            _unlock_btns, None, gen_btns)

        _N = len(poses.POSES)

        def _rerun_poses(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                         resolution, loras, lora_mult, pos, neg, sel_base,
                         face_mode, body_mode, face_ref, body_ref, body_ip_scale,
                         body_denoise, pose_face_adet, pfa_pos, pfa_neg,
                         pose_body_adet, pba_pos, pba_neg, pose_sdxl, *rest,
                         progress=gr.Progress()):
            cur = list(rest[:_N])
            choices = list(rest[_N:2 * _N])
            colors = list(rest[2 * _N:3 * _N])
            pstate = rest[3 * _N] if len(rest) > 3 * _N else {}
            if not any(cur):
                raise gr.Error("Generate poses first.")
            backend, ident = discovery.parse_model_value(model)
            sdxl_ident = ident if backend == "sd" else (pose_sdxl or "")  # see _gen_poses
            face_src = sel_base if face_mode == "Use Base" else (
                face_ref if face_mode == "Use Reference" else None)
            body_src = sel_base if body_mode == "Use Base" else (
                body_ref if body_mode == "Use Reference" else None)
            apply_face = bool(face_src)
            do_body = bool(body_src) and bool(sdxl_ident)
            if bool(body_src) and not sdxl_ident:
                gr.Warning("Body double needs an SDXL checkpoint — pick one in "
                           "'Body-swap / detail model' — skipping body double.")
            adet = _pose_adet(pose_face_adet, pfa_pos, pfa_neg,
                              pose_body_adet, pba_pos, pba_neg, sdxl_ident)
            if do_body:
                self._require(models.BODY_SWAP_KEYS, "Body double for poses")
            if apply_face:
                self._require(["inswapper_128", "buffalo_l"], "Pose face swap")
            specs = (pstate or {}).get("specs", [])
            P = poses.POSES
            base_clean = character.strip_base_framing(pos)  # see _gen_poses
            prev_snapshot = self._snapshot_prev(cur)  # for the per-pose ↩ undo
            final = list(cur)
            display = list(cur)  # streamed to the gallery as each re-roll lands
            to_swap = []  # (orig_index, new_base_img, spec)
            gen_sd.clear_abort()  # fresh abort state for this batch
            for i, img in enumerate(cur):
                if gen_sd.was_aborted():  # master abort → stop re-running the rest
                    break
                choice = choices[i] if i < len(choices) else "Approve"
                if choice == "Approve" or not img:
                    continue  # keep as-is (already swapped) / nothing to reroll
                if gen_sd.should_skip(i):  # this pose individually aborted → keep as-is
                    continue
                if choice == "Sharpen (no upscale)":  # whole-image crisp, no model/regen
                    final[i] = gen_sd.sharpen(img)
                    display[i] = final[i]
                    yield list(display)
                    continue
                gen_sd.set_current_index(i)
                # Use THIS slot's pose (index i ↔ P[i]) — NOT a distance+angle lookup,
                # which collided (many poses share full/front) and re-rolled sitting/
                # kneeling/close poses as the first standing match.
                ps = P[i] if i < len(P) else None
                spec = (specs[i] if i < len(specs)
                        else ({"distance": ps.distance, "angle": ps.angle,
                               "orientation": ps.orientation} if ps else
                              {"distance": "full", "angle": "front", "orientation": "portrait"}))
                desc = ps.description if ps else ""
                p_pos = ", ".join(p for p in (base_clean, desc) if p)
                p_neg = poses.pose_negative_for(
                    (ps.distance if ps else spec.get("distance", "full")), neg)
                orientation = ps.orientation if ps else spec.get("orientation", "portrait")
                pw, ph = self._res(resolution, self._orient_for(model, orientation))
                sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                img2img = choice in ("Cohesion (img2img)", "Re-Roll (img2img)")
                if img2img and backend == "native":
                    # Native img2img through Wan2GP's queue when the model has a
                    # guide-image input; else fall back to a fresh txt2img.
                    i2i_den = 0.35 if choice == "Cohesion (img2img)" else 0.6
                    if self._native_caps(ident).get("inpaint_support"):
                        imgs = self._gen_native(ident, p_pos, p_neg, pw, ph, steps,
                                                cfg, sd, mode="img2img",
                                                denoise=i2i_den, guide_path=img,
                                                loras=loras, mult_str=lora_mult, api=api)
                    else:
                        imgs = self._gen_image(state, model, p_pos, p_neg, pw, ph,
                                               steps, cfg, sd, sampler, scheduler,
                                               clip_skip, loras=loras, lora_mult=lora_mult,
                                               api=api)
                    new = imgs[0] if imgs else img
                elif choice == "Regenerate (txt2img)":
                    imgs = self._gen_image(state, model, p_pos, p_neg, pw, ph, steps,
                                           cfg, sd, sampler, scheduler, clip_skip,
                                           loras=loras, lora_mult=lora_mult, api=api)
                    new = imgs[0] if imgs else img
                else:  # SD img2img — Cohesion = gentle low-CFG; Re-Roll = heavier
                    if choice == "Cohesion (img2img)":
                        i2i_cfg, i2i_steps, i2i_den = 0.22, 14, 0.35
                    else:  # Re-Roll
                        i2i_cfg, i2i_steps, i2i_den = float(cfg), 24, 0.6
                    gen_sd.release_sd(); self._release_faceswap()
                    if self.acquire_gpu(state):
                        try:
                            outs = gen_sd.generate_img2img(
                                ident, img, p_pos, p_neg, pw, ph, i2i_steps,
                                i2i_cfg, sd, denoise=i2i_den, clip_skip=int(clip_skip),
                                loras=self._sd_lora_list(loras, lora_mult))
                        finally:
                            self.release_gpu(state)
                        new = outs[0] if outs else img
                    else:
                        new = img
                if gen_sd.should_skip(i):  # interrupted mid-gen → keep the original
                    continue
                to_swap.append((i, new, spec))
                display[i] = new  # show the re-rolled image now; swap pass refines it
                yield list(display)
            # On a master abort, skip the (slow) swap pass but KEEP the bare re-rolled
            # images already produced (don't revert them to the pre-run version).
            if to_swap and not gen_sd.was_aborted():
                items = [(im, sp) for (_, im, sp) in to_swap]
                swapped, _ = self._pose_swaps(
                    state, sdxl_ident, items, pos, neg, do_body, body_src, body_ip_scale,
                    body_denoise, apply_face, face_src, paths.cache_dir() / "poses",
                    progress, start_idx=1000, adet=adet)
                # Color match (only for ticked Cohesion/Re-Roll poses) → base tones.
                # It runs the person-seg YOLO on the GPU, so hold the shared lock
                # (only acquire if any pose actually needs it).
                need_cm = bool(sel_base) and any(
                    oi < len(colors) and colors[oi] for (oi, _, _) in to_swap)
                cm_locked = need_cm and self.acquire_gpu(state)
                try:
                    for (oi, _, _), sw in zip(to_swap, swapped):
                        if cm_locked and sw and oi < len(colors) and colors[oi] and sel_base:
                            sw = gen_sd.color_match(sw, sel_base, body_only=True)
                        final[oi] = sw
                finally:
                    if cm_locked:
                        self.release_gpu(state)
            elif to_swap:  # aborted → keep the un-swapped re-rolls (mirrors _gen_poses)
                for (oi, im, _) in to_swap:
                    final[oi] = im
            saved = self._persist_poses(final, specs)
            return saved + [{"poses": saved, "specs": specs}, prev_snapshot]

        pose["rerun"].click(_lock_btns, None, gen_btns).then(
            _rerun_poses,
            inputs=pose_in + pose["pose_imgs"] + pose["pose_choices"] + pose["pose_color"]
            + [ui["poses_state"]],
            outputs=pose_out + [pose["pose_prev"]]).then(_unlock_btns, None, gen_btns)

        # -- Repass Face / Body on Selected: re-run ONLY that swap pass on the poses
        # whose "Face/Body Repass" tickbox is on. Unlike Re-run, this ignores the
        # Approve dropdown — an Approve'd pose CAN be repassed if it's ticked. --
        def _do_repass(kind, state, model, pos, neg, sel_base, face_mode, body_mode,
                       face_ref, body_ref, body_ip_scale, body_denoise,
                       pfa, pfa_pos, pfa_neg, pba, pba_pos, pba_neg, pose_sdxl,
                       *rest, progress=gr.Progress()):
            cur = list(rest[:_N])
            flags = list(rest[_N:2 * _N])  # per-pose repass tickboxes
            pstate = rest[2 * _N] if len(rest) > 2 * _N else {}
            sel = [i for i in range(min(_N, len(cur)))
                   if i < len(flags) and flags[i] and cur[i]]
            if not sel:
                raise gr.Error("Tick “Face/Body Repass” on at least one pose first.")
            backend, ident = discovery.parse_model_value(model)
            sdxl_ident = ident if backend == "sd" else (pose_sdxl or "")  # see _gen_poses
            face_src = sel_base if face_mode == "Use Base" else (
                face_ref if face_mode == "Use Reference" else None)
            body_src = sel_base if body_mode == "Use Base" else (
                body_ref if body_mode == "Use Reference" else None)
            specs = (pstate or {}).get("specs", [])
            if kind == "face":
                if not face_src:
                    raise gr.Error("Set “Face swap” to Use Base/Reference first.")
                self._require(["inswapper_128", "buffalo_l"], "Face repass")
                adet = _pose_adet(pfa, pfa_pos, pfa_neg, False, "", "", sdxl_ident)
                do_body, apply_face = False, True
            else:  # body
                if not body_src:
                    raise gr.Error("Set “Body double” to Use Base/Reference first.")
                if not sdxl_ident:
                    raise gr.Error("Pick a “Body-swap / detail model (SDXL)” first.")
                self._require(models.BODY_SWAP_KEYS, "Body repass")
                adet = _pose_adet(False, "", "", pba, pba_pos, pba_neg, sdxl_ident)
                do_body, apply_face = True, False
            prev_snapshot = self._snapshot_prev(cur)  # for the per-pose ↩ undo
            gen_sd.clear_abort()
            items = [(cur[i], (specs[i] if i < len(specs) else {})) for i in sel]
            # start_idx=2000 keeps repass output filenames clear of generate (1..) and
            # re-run (1000..), so it never clobbers another slot's file.
            swapped, _ = self._pose_swaps(
                state, sdxl_ident, items, pos, neg, do_body, body_src,
                body_ip_scale, body_denoise, apply_face, face_src,
                paths.cache_dir() / "poses", progress, start_idx=2000, adet=adet)
            final = list(cur)
            for j, i in enumerate(sel):
                if j < len(swapped) and swapped[j]:
                    final[i] = swapped[j]
            saved = self._persist_poses(final, specs)
            return saved + [{"poses": saved, "specs": specs}, prev_snapshot]

        def _repass_face(*args, progress=gr.Progress()):
            return _do_repass("face", *args, progress=progress)

        def _repass_body(*args, progress=gr.Progress()):
            return _do_repass("body", *args, progress=progress)

        repass_in = [self.state, SET[0], prm["positive_prompt"],
                     prm["negative_prompt"], base["selected_base"], pose["face_mode"],
                     pose["body_mode"], swap["face_source"], swap["body_source"],
                     swap["body_ip_scale"], swap["body_denoise"],
                     pose["pose_face_adet"], pose["pose_face_adet_pos"],
                     pose["pose_face_adet_neg"], pose["pose_body_adet"],
                     pose["pose_body_adet_pos"], pose["pose_body_adet_neg"],
                     pose["pose_sdxl_model"]] + pose["pose_imgs"] \
            + pose["pose_repass"] + [ui["poses_state"]]
        repass_out = pose_out + [pose["pose_prev"]]

        pose["repass_face"].click(_lock_btns, None, gen_btns).then(
            _repass_face, inputs=repass_in, outputs=repass_out).then(
            _unlock_btns, None, gen_btns)
        pose["repass_body"].click(_lock_btns, None, gen_btns).then(
            _repass_body, inputs=repass_in, outputs=repass_out).then(
            _unlock_btns, None, gen_btns)

        # Per-pose ↩ undo: revert one slot to its pre-re-run image (+ sync state +
        # re-persist so the revert survives an app reload).
        def _make_pose_undo(i):
            def _h(prev, pstate):
                if not prev or i >= len(prev) or not prev[i]:
                    return gr.update(), gr.update()
                p = dict(pstate or {})
                gallery = list(p.get("poses", []))
                while len(gallery) <= i:
                    gallery.append(None)
                gallery[i] = prev[i]
                specs = p.get("specs", [])
                saved = self._persist_poses(gallery, specs)  # rewrite canonical + state
                p["poses"] = saved
                return (saved[i] if i < len(saved) else prev[i]), p
            return _h
        for _i, _ub in enumerate(pose["pose_undo"]):
            _ub.click(_make_pose_undo(_i),
                      inputs=[pose["pose_prev"], ui["poses_state"]],
                      outputs=[pose["pose_imgs"][_i], ui["poses_state"]])

        # -- Replicate base thumbnail: mirrors the current base; ➕ on a pose sets it
        # as the new base (saving the old one), and ↩ Undo base reverts that. --
        base_thumb = pose.get("pose_base_thumb")
        base_prev = pose.get("pose_base_prev")
        if base_thumb is not None:
            # Keep the thumbnail in sync whenever the base changes (anywhere).
            base["selected_base"].change(lambda b: b or None,
                                         inputs=[base["selected_base"]],
                                         outputs=[base_thumb])

            def _set_as_base(pose_path, cur_base):
                if not pose_path:
                    gr.Warning("That pose slot is empty.")
                    return gr.update(), gr.update()
                gr.Info("Set as base. Use ↩ Undo base to revert.")
                return pose_path, cur_base  # selected_base ← pose, prev ← old base

            for _i, _sb in enumerate(pose["pose_setbase"]):
                _sb.click(_set_as_base,
                          inputs=[pose["pose_imgs"][_i], base["selected_base"]],
                          outputs=[base["selected_base"], base_prev])

            def _undo_base(prev):
                if not prev:
                    gr.Info("No previous base to revert to.")
                    return gr.update()
                return prev  # selected_base ← previous (thumb follows via .change)

            if pose.get("pose_base_undo") is not None:
                pose["pose_base_undo"].click(_undo_base, inputs=[base_prev],
                                             outputs=[base["selected_base"]])

        # "Use Reference" is only offered when a face/body reference exists on Human Clone.
        def _ref_opts(src, base):
            opts = ["None", "Use Base"] + (["Use Reference"] if src else [])
            return gr.update(choices=opts)
        swap["face_source"].change(_ref_opts, inputs=[swap["face_source"], base["selected_base"]],
                                   outputs=[pose["face_mode"]])
        swap["body_source"].change(_ref_opts, inputs=[swap["body_source"], base["selected_base"]],
                                   outputs=[pose["body_mode"]])

    @staticmethod
    def _ctx_path_allowed(p) -> bool:
        """Confine right-clicked file refs to the app's own roots so a crafted relay
        value can't turn the 'Reference' action into an arbitrary-file read."""
        import os
        import tempfile
        try:
            rp = os.path.realpath(p)
        except Exception:
            return False
        roots = []
        for r in (paths.lab_root(), paths.cache_dir(), paths.orphansuite_root(),
                  Path(os.getcwd()), Path(tempfile.gettempdir())):
            try:
                roots.append(os.path.realpath(r))
            except Exception:
                pass
        for r in roots:
            try:
                if os.path.commonpath([rp, r]) == r:
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _resolve_ctx_src(src):
        """Resolve a right-clicked media src to a local file path: a /file= URL → its
        path; a data: URL → decoded to the persist dir. Returns a path or None.
        File refs are confined to the app's roots; data URLs are size-capped and
        validated as real images before use."""
        import base64
        import io
        import os
        import time
        import urllib.parse
        if not src:
            return None
        if src.startswith("data:"):
            try:
                head, b64 = src.split(",", 1)
                if len(b64) > 64 * 1024 * 1024:  # ~48MB decoded — reject huge payloads
                    return None
                raw = base64.b64decode(b64)
                from PIL import Image
                Image.open(io.BytesIO(raw)).verify()  # reject non-image / bomb headers
                ext = ".jpg" if "jpeg" in head or "jpg" in head else ".png"
                d = paths.cache_dir() / "persist"
                d.mkdir(parents=True, exist_ok=True)
                f = d / f"ctxref_{int(time.time() * 1000)}{ext}"
                f.write_bytes(raw)
                return str(f)
            except Exception:
                return None
        if "/file=" in src:
            p = urllib.parse.unquote(src.split("/file=", 1)[1].split("?", 1)[0])
            return p if (os.path.isfile(p) and ReplicantCharLab._ctx_path_allowed(p)) else None
        return src if (os.path.isfile(src) and ReplicantCharLab._ctx_path_allowed(src)) else None

    def _wire_context_menu(self, ui):
        """Right-click → 'Replicant (Reference)': load the image as the Setup reference
        and switch to our tab."""
        relay = ui["components"]["setup"].get("ctx_relay")
        reference = ui["components"]["setup"].get("reference_image")
        if relay is None or reference is None:
            return
        tabs = getattr(self, "main_tabs", None)

        def _to_reference(val):
            src = (val or "").rsplit("|", 1)[0]
            path = self._resolve_ctx_src(src)
            if not path:
                gr.Warning("Couldn't read that image for Replicant.")
                return (gr.update(), gr.update()) if tabs is not None else gr.update()
            gr.Info("Loaded into Replicant as the reference image.")
            if tabs is not None:
                return path, gr.Tabs(selected=PLUGIN_ID)  # set reference + switch tab
            return path

        outs = [reference, tabs] if tabs is not None else [reference]
        relay.change(_to_reference, inputs=[relay], outputs=outs)

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

        # Cohesion subtab uses the same enhancer for its own prompts.
        inp = ui["components"]["inpaint"]
        inp["cohesion_enhance_pos"].click(
            _enhance, inputs=[self.state, inp["cohesion_prompt"]],
            outputs=[inp["cohesion_prompt"]])
        inp["cohesion_enhance_neg"].click(
            _enhance, inputs=[self.state, inp["cohesion_neg"]],
            outputs=[inp["cohesion_neg"]])


Plugin = ReplicantCharLab
