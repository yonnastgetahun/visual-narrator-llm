"use client";

import type { BufferGeometry, LineBasicMaterial, LineSegments, Points, PointsMaterial, WebGLRenderer } from "three";
import { useEffect, useRef } from "react";

function getCurrentTheme(): "dark" | "light" {
  if (typeof document === "undefined") return "dark";
  return (document.documentElement.dataset.theme as "dark" | "light") ?? "dark";
}

/**
 * FilmFrameCanvas — Three.js hero element.
 * 12–15 film frames floating in 3D depth with depth-of-field dimming,
 * dense amber film-grain particles, multi-layer glowing waveform,
 * and a slow cinematic camera push.
 * Transparent background, composites over the dark hero section.
 * Strictly isolated — no state, no parent rerenders.
 */
export function FilmFrameCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let animId: number;
    let cleanedUp = false;

    async function init() {
      const THREE = await import("three");

      if (cleanedUp || !canvas) return;

      const W = canvas.clientWidth;
      const H = canvas.clientHeight;

      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
      renderer.setSize(W, H, false);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      function applyThemeToRenderer(r: WebGLRenderer) {
        const theme = getCurrentTheme();
        r.setClearColor(theme === "light" ? 0xF7F5F0 : 0x0A0A0A, 1);
      }
      applyThemeToRenderer(renderer);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(58, W / H, 0.1, 80);
      camera.position.set(0, 0, 6);

      // ─── Film frame helper ───────────────────────────────────────
      function makeFrameGeo(w: number, h: number) {
        const hw = w / 2, hh = h / 2;
        const d = 0.045;
        const pts: number[] = [
          // outer rect
          -hw, -hh, 0,  hw, -hh, 0,
           hw, -hh, 0,  hw,  hh, 0,
           hw,  hh, 0, -hw,  hh, 0,
          -hw,  hh, 0, -hw, -hh, 0,
          // corner marks × 4 × 2 lines
          -hw+d, -hh+d*0.3, 0,  -hw+d, -hh+d*1.5, 0,
          -hw+d, -hh+d*0.3, 0,  -hw+d*2.4, -hh+d*0.3, 0,
           hw-d,  hh-d*0.3, 0,   hw-d,  hh-d*1.5, 0,
           hw-d,  hh-d*0.3, 0,   hw-d*2.4,  hh-d*0.3, 0,
          -hw+d,  hh-d*0.3, 0,  -hw+d,  hh-d*1.5, 0,
          -hw+d,  hh-d*0.3, 0,  -hw+d*2.4,  hh-d*0.3, 0,
           hw-d, -hh+d*0.3, 0,   hw-d, -hh+d*1.5, 0,
           hw-d, -hh+d*0.3, 0,   hw-d*2.4, -hh+d*0.3, 0,
        ];
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
        return geo;
      }

      // Depth-of-field: brightness tied to z position
      // z range roughly -1.0 (foreground, bright) to -7.5 (background, very dim)
      function depthOpacity(z: number, baseOpacity: number): number {
        // foreground z = -1: factor ~1.0 | deep z = -7: factor ~0.18
        const near = -1.0;
        const far  = -7.5;
        const t = Math.max(0, Math.min(1, (z - near) / (far - near)));
        return baseOpacity * (1.0 - t * 0.82);
      }

      // Material factory — each frame gets its own material for per-frame opacity
      function makeMat(color: number, opacity: number) {
        return new THREE.LineBasicMaterial({ color, transparent: true, opacity });
      }

      const geoStd   = makeFrameGeo(2.0, 1.35);
      const geoWide  = makeFrameGeo(2.6, 1.55);
      const geoWider = makeFrameGeo(3.1, 1.75);
      const geoSmall = makeFrameGeo(1.4, 1.0);
      const geoTiny  = makeFrameGeo(1.1, 0.78);

      type FrameConfig = {
        geo: BufferGeometry;
        color: number;
        baseOpacity: number;
        x: number; y: number; z: number;
        rx: number; ry: number; rz: number;
        scale: number;
        speed: number;
        phase: number;
        driftX: number;
      };

      // 14 frames — varied z for depth layering
      const configs: FrameConfig[] = [
        // --- Foreground (sharp, bright amber) ---
        { geo: geoStd,   color: 0xF59E0B, baseOpacity: 0.94, x: -3.0,  y:  0.6,  z: -1.0, rx: -0.04, ry:  0.16, rz:  0.03, scale: 1.05, speed: 0.0028, phase: 0.0,  driftX:  0.12 },
        { geo: geoSmall, color: 0xF59E0B, baseOpacity: 0.88, x:  2.2,  y: -1.2,  z: -1.5, rx: -0.03, ry:  0.28, rz:  0.02, scale: 0.82, speed: 0.0045, phase: 3.1,  driftX: -0.08 },
        { geo: geoTiny,  color: 0xD4D0C8, baseOpacity: 0.82, x: -1.2,  y:  2.1,  z: -1.8, rx:  0.06, ry: -0.20, rz:  0.05, scale: 0.92, speed: 0.0038, phase: 1.5,  driftX:  0.06 },
        // --- Midground (dimmer cream/amber) ---
        { geo: geoWide,  color: 0xD4D0C8, baseOpacity: 0.64, x:  0.3,  y: -0.4,  z: -3.2, rx:  0.05, ry: -0.08, rz: -0.02, scale: 1.28, speed: 0.0019, phase: 1.1,  driftX: -0.10 },
        { geo: geoStd,   color: 0xD4D0C8, baseOpacity: 0.58, x:  3.1,  y:  0.8,  z: -2.8, rx:  0.03, ry: -0.22, rz:  0.05, scale: 0.95, speed: 0.0035, phase: 2.3,  driftX:  0.07 },
        { geo: geoSmall, color: 0xF59E0B, baseOpacity: 0.52, x: -3.8,  y: -0.5,  z: -3.6, rx:  0.08, ry:  0.14, rz: -0.03, scale: 1.10, speed: 0.0024, phase: 4.7,  driftX: -0.05 },
        { geo: geoWide,  color: 0xD4D0C8, baseOpacity: 0.46, x:  1.8,  y:  1.4,  z: -4.0, rx: -0.06, ry:  0.18, rz:  0.04, scale: 1.20, speed: 0.0030, phase: 2.9,  driftX:  0.09 },
        { geo: geoTiny,  color: 0xA8A8A8, baseOpacity: 0.42, x: -0.8,  y: -2.0,  z: -3.4, rx:  0.10, ry: -0.12, rz:  0.03, scale: 0.72, speed: 0.0041, phase: 5.2,  driftX: -0.07 },
        // --- Background (ghost, very dim) ---
        { geo: geoWider, color: 0xA8A8A8, baseOpacity: 0.28, x: -1.6,  y: -1.6,  z: -5.5, rx:  0.07, ry:  0.11, rz: -0.04, scale: 1.60, speed: 0.0014, phase: 0.7,  driftX:  0.04 },
        { geo: geoStd,   color: 0xA8A8A8, baseOpacity: 0.22, x: -4.4,  y: -0.3,  z: -4.8, rx:  0.09, ry: -0.15, rz:  0.03, scale: 1.40, speed: 0.0018, phase: 1.9,  driftX: -0.06 },
        { geo: geoWide,  color: 0x8B8070, baseOpacity: 0.18, x:  4.0,  y:  0.2,  z: -6.2, rx: -0.07, ry:  0.06, rz: -0.03, scale: 1.75, speed: 0.0011, phase: 4.2,  driftX:  0.03 },
        { geo: geoWider, color: 0x8B8070, baseOpacity: 0.14, x:  0.8,  y:  2.5,  z: -7.0, rx:  0.04, ry: -0.09, rz:  0.02, scale: 1.90, speed: 0.0009, phase: 3.8,  driftX: -0.04 },
        { geo: geoSmall, color: 0x8B8070, baseOpacity: 0.12, x: -2.6,  y:  1.2,  z: -6.6, rx:  0.11, ry:  0.07, rz: -0.05, scale: 1.15, speed: 0.0013, phase: 6.1,  driftX:  0.05 },
        { geo: geoTiny,  color: 0x6B6050, baseOpacity: 0.10, x:  3.4,  y: -1.8,  z: -7.4, rx: -0.08, ry: -0.04, rz:  0.03, scale: 0.88, speed: 0.0016, phase: 2.4,  driftX: -0.03 },
      ];

      type FrameObj = {
        mesh: LineSegments;
        mat: LineBasicMaterial;
        basePosY: number;
        speed: number;
        phase: number;
        driftX: number;
        basePosX: number;
      };

      const frames: FrameObj[] = configs.map((cfg) => {
        const opacity = depthOpacity(cfg.z, cfg.baseOpacity);
        const mat = makeMat(cfg.color, opacity);
        const mesh = new THREE.LineSegments(cfg.geo, mat);
        mesh.position.set(cfg.x, cfg.y, cfg.z);
        mesh.rotation.set(cfg.rx, cfg.ry, cfg.rz);
        mesh.scale.setScalar(cfg.scale);
        scene.add(mesh);
        return { mesh, mat, basePosY: cfg.y, basePosX: cfg.x, speed: cfg.speed, phase: cfg.phase, driftX: cfg.driftX };
      });

      // ─── Dust particles — dense amber film grain ─────────────────
      // Three layers for parallax depth feel
      const layers = [
        { count: 320, spread: [16, 9, 9], offset: -2,  size: 0.028, opacity: 0.34 },
        { count: 200, spread: [12, 7, 7], offset: -4,  size: 0.018, opacity: 0.20 },
        { count: 140, spread: [10, 6, 5], offset: -6,  size: 0.012, opacity: 0.12 },
      ];

      type ParticleLayer = { pts: Points; rotSpeedY: number; rotSpeedX: number };
      const particleLayers: ParticleLayer[] = layers.map((l) => {
        const N = l.count;
        const pos = new Float32Array(N * 3);
        for (let i = 0; i < N; i++) {
          pos[i * 3]     = (Math.random() - 0.5) * l.spread[0];
          pos[i * 3 + 1] = (Math.random() - 0.5) * l.spread[1];
          pos[i * 3 + 2] = (Math.random() - 0.5) * l.spread[2] + l.offset;
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
        const mat = new THREE.PointsMaterial({
          color: 0xF59E0B,
          size: l.size,
          transparent: true,
          opacity: l.opacity,
          sizeAttenuation: true,
        });
        const pts = new THREE.Points(geo, mat);
        scene.add(pts);
        return { pts, rotSpeedY: 0.008 + Math.random() * 0.006, rotSpeedX: 0.004 + Math.random() * 0.003 };
      });

      // ─── Waveform — multi-layer amber glow ───────────────────────
      const WN = 240;

      // Build raw wave values into a shared buffer
      const wPos0 = new Float32Array(WN * 3);
      const wPos1 = new Float32Array(WN * 3);
      const wPos2 = new Float32Array(WN * 3);

      function buildWavePositions(buf: Float32Array, t: number, yBase: number) {
        for (let i = 0; i < WN; i++) {
          const x = (i / (WN - 1)) * 8.0 - 4.0;
          const y =
            Math.sin(i * 0.15 + t * 1.7) * 0.28 +
            Math.sin(i * 0.28 + t * 2.5) * 0.13 +
            Math.sin(i * 0.07 + t * 0.8) * 0.18 +
            Math.sin(i * 0.52 + t * 3.1) * 0.06;
          buf[i * 3]     = x;
          buf[i * 3 + 1] = y + yBase;
          buf[i * 3 + 2] = -0.8;
        }
      }

      // Layer 0: outer glow — widest, most transparent
      const wGeo0 = new THREE.BufferGeometry();
      const wMat0 = new THREE.LineBasicMaterial({ color: 0xF59E0B, transparent: true, opacity: 0.18 });
      const waveLine0 = new THREE.Line(wGeo0, wMat0);
      scene.add(waveLine0);

      // Layer 1: mid glow
      const wGeo1 = new THREE.BufferGeometry();
      const wMat1 = new THREE.LineBasicMaterial({ color: 0xF59E0B, transparent: true, opacity: 0.44 });
      const waveLine1 = new THREE.Line(wGeo1, wMat1);
      scene.add(waveLine1);

      // Layer 2: core — full opacity bright amber
      const wGeo2 = new THREE.BufferGeometry();
      const wMat2 = new THREE.LineBasicMaterial({ color: 0xFFD580, transparent: true, opacity: 0.85 });
      const waveLine2 = new THREE.Line(wGeo2, wMat2);
      scene.add(waveLine2);

      function updateWave(t: number) {
        const yBase = -2.6;
        buildWavePositions(wPos0, t, yBase);
        buildWavePositions(wPos1, t, yBase);
        buildWavePositions(wPos2, t, yBase);

        // Offset layers slightly vertically for thickness illusion
        for (let i = 0; i < WN; i++) {
          wPos0[i * 3 + 1] += 0.045;
          wPos1[i * 3 + 1] += 0.018;
          // wPos2 stays at exact base — the sharp core
        }

        wGeo0.setAttribute("position", new THREE.Float32BufferAttribute(wPos0, 3));
        wGeo1.setAttribute("position", new THREE.Float32BufferAttribute(wPos1, 3));
        wGeo2.setAttribute("position", new THREE.Float32BufferAttribute(wPos2, 3));
        wGeo0.attributes.position.needsUpdate = true;
        wGeo1.attributes.position.needsUpdate = true;
        wGeo2.attributes.position.needsUpdate = true;
      }
      updateWave(0);

      // ─── Animation loop ──────────────────────────────────────────
      let t = 0;
      function animate() {
        if (cleanedUp) return;
        animId = requestAnimationFrame(animate);
        t += 0.016;

        // Slow cinematic camera push — slightly more pronounced drift
        camera.position.x = Math.sin(t * 0.055) * 0.28;
        camera.position.y = Math.cos(t * 0.038) * 0.16;
        // Subtle push forward/back (z oscillation)
        camera.position.z = 6.0 + Math.sin(t * 0.022) * 0.35;

        frames.forEach((f) => {
          f.mesh.position.y = f.basePosY + Math.sin(t * f.speed * 55 + f.phase) * 0.16;
          f.mesh.position.x = f.basePosX + Math.sin(t * f.speed * 30 + f.phase * 0.7) * f.driftX;
          f.mesh.rotation.y += f.speed * 0.55;
        });

        // Each particle layer rotates at its own speed (parallax)
        const isLight = getCurrentTheme() === "light";
        particleLayers.forEach((l, idx) => {
          l.pts.rotation.y = t * l.rotSpeedY;
          l.pts.rotation.x = t * l.rotSpeedX;
          // Flicker: randomize opacity very slightly each frame for grain feel
          // Reduce opacity slightly in light mode — amber reads differently on paper
          if (idx === 0) {
            const base = isLight ? 0.20 : 0.30;
            (l.pts.material as PointsMaterial).opacity = base + Math.random() * 0.08;
          }
        });

        updateWave(t);
        renderer.render(scene, camera);
      }
      animate();

      // ─── Theme observer — reacts to data-theme changes ───────────
      const themeObserver = new MutationObserver(() => {
        applyThemeToRenderer(renderer);
      });
      themeObserver.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-theme"],
      });

      // ─── Resize ──────────────────────────────────────────────────
      const ro = new ResizeObserver(() => {
        if (!canvas) return;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      });
      ro.observe(canvas);

      // Return cleanup fn
      return () => {
        themeObserver.disconnect();
        ro.disconnect();
        renderer.dispose();
        geoStd.dispose();
        geoWide.dispose();
        geoWider.dispose();
        geoSmall.dispose();
        geoTiny.dispose();
        particleLayers.forEach((l) => {
          l.pts.geometry.dispose();
          (l.pts.material as PointsMaterial).dispose();
        });
        [wGeo0, wGeo1, wGeo2].forEach((g) => g.dispose());
        [wMat0, wMat1, wMat2].forEach((m) => m.dispose());
        frames.forEach((f) => f.mat.dispose());
      };
    }

    let disposeScene: (() => void) | undefined;
    init().then((cleanup) => { disposeScene = cleanup; });

    return () => {
      cleanedUp = true;
      cancelAnimationFrame(animId);
      disposeScene?.();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="h-full w-full"
      style={{ display: "block" }}
    />
  );
}
