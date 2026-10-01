import * as THREE from "three";
import { STLLoader } from "./vendor/STLLoader.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

// 3D fly brain: the real male-cns:v1.0 brain surface (frontend/brain.stl,
// built by scripts/build_brain_3d.py) as a translucent shell, with one dot
// per circuit neuron at its real position. Held in a fixed front view.
//
// Thinking animation (play): the decision's real spike events -- per neuron
// a first-spike time and a spike count -- are laid on a timeline stretched to
// the fly's whole thinking pause and advanced every frame. Each neuron eases
// up, then decays slowly and leaves a faint afterglow, so activity reads as a
// wave spreading through the network rather than isolated blinks. Light
// packets travel along real (sampled) synapses between neurons that fired in
// order, and when the decision is revealed the output neurons flare.

const RISE_MS = 240;      // ease-in time of a neuron lighting up
const DECAY_TAU_MS = 1000; // e-fold time of its glow after the peak
const TRAIL_TAU_MS = 2600;
const INTENSITY = 0.55;   // overall brightness of spike glow (lower = dimmer)
const REVEAL_FLARE = 0.8; // how brightly the output neurons flare at the decision
const SIM_MS = 150;       // length of one brain simulation run (agents.py N_SIM_STEPS)
const MAX_PACKETS = 320;

const smoothstep = (x) => {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
};

function init(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000);
  camera.position.set(0, 4, 150);

  scene.add(new THREE.AmbientLight(0x8fa4d8, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(60, 90, 120);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x6f9bff, 0.9);
  rim.position.set(-80, -30, -100);
  scene.add(rim);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.enableRotate = false;
  controls.autoRotate = false;

  const brain = new THREE.Group();
  brain.rotation.x = Math.PI; // neuPrint's y axis points down; flip to upright
  scene.add(brain);

  let shell = null;
  let shellPulse = 0;
  new STLLoader().load("brain.stl", (geometry) => {
    geometry.computeVertexNormals();
    shell = new THREE.Mesh(
      geometry,
      new THREE.MeshPhongMaterial({
        color: 0x6f8fe8,
        transparent: true,
        opacity: 0.22,
        shininess: 60,
        depthWrite: false,
        side: THREE.DoubleSide,
      })
    );
    brain.add(shell);
  });

  // ---- neuron dots -------------------------------------------------------
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { scale: { value: 1 }, time: { value: 0 }, awake: { value: 0 } },
    vertexShader: `
      attribute float glow;
      attribute float trail;
      attribute float baseSize;
      attribute float phase;
      attribute vec3 baseColor;
      varying vec3 vColor;
      varying float vGlow;
      varying float vBreath;
      uniform float scale;
      uniform float time;
      uniform float awake;
      void main() {
        float g = clamp(glow, 0.0, 1.5);
        // resting colour -> orange -> hot yellow-white as it lights up
        vec3 warm = mix(baseColor * (1.0 + trail * 1.6), vec3(0.9, 0.5, 0.18), smoothstep(0.0, 0.5, g));
        vColor = mix(warm, vec3(0.92, 0.78, 0.42), smoothstep(0.35, 0.9, g));
        vGlow = g;
        vBreath = (0.86 + 0.14 * sin(time * 0.9 + phase * 6.2831)) * (1.0 + awake * 0.45);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = (baseSize + g * 5.0 + trail * 1.5) * scale * (150.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      varying float vGlow;
      varying float vBreath;
      void main() {
        float d = length(gl_PointCoord - vec2(0.5)) * 2.0;
        if (d > 1.0) discard;
        float core = (1.0 - d) * 0.42 * vBreath;
        float halo = exp(-d * d * 3.2) * vGlow * 0.5;   // soft bloom-like glow
        gl_FragColor = vec4(vColor, core + halo);
      }`,
  });

  let points = null;
  let n = 0;
  let glow, trail, t0, peak;
  let slotOf = new Map();
  let outSlots = [];
  let stageOf = null; // 0 = input population, 1 = relay, 2 = output
  let edgeSlots = [];
  const active = new Set();

  function setNodes(nodes, roleColors, edges) {
    if (points) {
      brain.remove(points);
      points.geometry.dispose();
    }
    n = nodes.length;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const phase = new Float32Array(n);
    glow = new Float32Array(n);
    trail = new Float32Array(n);
    t0 = new Float64Array(n).fill(-1);
    peak = new Float32Array(n);
    slotOf = new Map();
    outSlots = [];
    stageOf = new Uint8Array(n);
    active.clear();
    const tmp = new THREE.Color();
    nodes.forEach((node, i) => {
      pos.set([node.x, node.y, node.z], i * 3);
      tmp.set(roleColors[node.role] || roleColors.relay || "#57628c");
      col.set([tmp.r, tmp.g, tmp.b], i * 3);
      size[i] = node.role === "relay" ? 1.5 : 2.6;
      phase[i] = Math.random();
      slotOf.set(node.index, i);
      stageOf[i] = node.role.startsWith("in:") ? 0 : node.role.startsWith("out:") ? 2 : 1;
      if (node.role === "out:action") outSlots.push(i);
    });
    edgeSlots = [];
    for (const [s, t] of edges || []) {
      const a = slotOf.get(s), b = slotOf.get(t);
      if (a !== undefined && b !== undefined) edgeSlots.push([a, b]);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute("baseColor", new THREE.BufferAttribute(col, 3));
    geometry.setAttribute("baseSize", new THREE.BufferAttribute(size, 1));
    geometry.setAttribute("phase", new THREE.BufferAttribute(phase, 1));
    geometry.setAttribute("glow", new THREE.BufferAttribute(glow, 1));
    geometry.setAttribute("trail", new THREE.BufferAttribute(trail, 1));
    points = new THREE.Points(geometry, material);
    points.frustumCulled = false;
    brain.add(points);
  }

  // ---- light packets along real synapses ---------------------------------
  const packetPos = new Float32Array(MAX_PACKETS * 3);
  const packetAlpha = new Float32Array(MAX_PACKETS);
  const packetGeometry = new THREE.BufferGeometry();
  packetGeometry.setAttribute("position", new THREE.BufferAttribute(packetPos, 3));
  packetGeometry.setAttribute("alpha", new THREE.BufferAttribute(packetAlpha, 1));
  const packetMaterial = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { scale: { value: 1 } },
    vertexShader: `
      attribute float alpha;
      varying float vAlpha;
      uniform float scale;
      void main() {
        vAlpha = alpha;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = 4.5 * scale * (150.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying float vAlpha;
      void main() {
        float d = length(gl_PointCoord - vec2(0.5)) * 2.0;
        if (d > 1.0) discard;
        gl_FragColor = vec4(1.0, 0.93, 0.7, exp(-d * d * 3.0) * vAlpha);
      }`,
  });
  const packetPoints = new THREE.Points(packetGeometry, packetMaterial);
  packetPoints.frustumCulled = false;
  brain.add(packetPoints);
  let packets = [];

  // ---- timeline ----------------------------------------------------------
  let events = [];
  let evPtr = 0;
  let playStart = 0;

  function trigger(slot, intensity, now) {
    peak[slot] = Math.max(intensity, active.has(slot) ? glow[slot] : 0);
    t0[slot] = now;
    active.add(slot);
  }

  function play(spikes, durationMs, now = performance.now()) {
    if (!points || !spikes || !spikes.length) return;
    // Most first-spikes land within a few ms of each other (90% inside the
    // first ~16 of 150 ms), so a linear time mapping would flash almost
    // everything at once and then just fade. Instead, order neurons by real
    // spike time -- ties broken input -> relay -> output, which is the
    // direction signals travel -- and spread them over the thinking pause by
    // rank (blended lightly with true time), so the wave visibly moves.
    const rows = [];
    for (const [idx, t, count] of spikes) {
      const slot = slotOf.get(idx);
      if (slot !== undefined) rows.push({ slot, t, count, stage: stageOf[slot] });
    }
    rows.sort((a, b) => a.t - b.t || a.stage - b.stage);
    const tMax = Math.max(SIM_MS * 0.25, rows[rows.length - 1].t);
    const span = durationMs; // spread over the whole pause so it never goes quiet
    events = [];
    const fireAt = new Map();
    const realT = new Map();
    rows.forEach((r, i) => {
      const rank = rows.length > 1 ? i / (rows.length - 1) : 0;
      const start = span * (0.2 * (r.t / tMax) + 0.8 * rank);
      fireAt.set(r.slot, start);
      realT.set(r.slot, r.t);
      events.push({ slot: r.slot, start, intensity: INTENSITY * (0.4 + 0.6 * Math.min(1, r.count / 6)) });
    });
    evPtr = 0;
    playStart = now;

    // A packet for each sampled synapse whose two ends fired in order.
    const candidates = [];
    for (const [a, b] of edgeSlots) {
      const ta = fireAt.get(a), tb = fireAt.get(b);
      if (ta !== undefined && tb !== undefined && tb > ta && realT.get(b) >= realT.get(a)) {
        candidates.push([a, b, ta, tb]);
      }
    }
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
    packets = candidates.slice(0, MAX_PACKETS).map(([a, b, ta, tb]) => ({
      a, b, start: ta, dur: Math.min(1100, Math.max(380, tb - ta)),
    }));
  }

  function reveal(now = performance.now()) {
    for (const slot of outSlots) trigger(slot, REVEAL_FLARE, now);
    shellPulse = 0.5;
  }

  let awakeTarget = 0;
  function setThinking(on) {
    awakeTarget = on ? 1 : 0;
  }

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight || w;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const px = renderer.getPixelRatio();
    material.uniforms.scale.value = px;
    packetMaterial.uniforms.scale.value = px;
  }
  new ResizeObserver(resize).observe(container);
  resize();

  // Advances all animation state to time `now` (ms). Split from the render
  // loop so tests can step simulated time deterministically.
  function update(now, dt) {
    material.uniforms.time.value = now / 1000;
    material.uniforms.awake.value += (awakeTarget - material.uniforms.awake.value) * (1 - Math.exp(-dt / 350));

    if (points) {
      const elapsed = now - playStart;
      while (evPtr < events.length && events[evPtr].start <= elapsed) {
        trigger(events[evPtr].slot, events[evPtr].intensity, now);
        evPtr++;
      }
      const trailDecay = Math.exp(-dt / TRAIL_TAU_MS);
      for (const slot of active) {
        const age = now - t0[slot];
        let level;
        if (age < RISE_MS) level = smoothstep(age / RISE_MS) * peak[slot];
        else level = peak[slot] * Math.exp(-(age - RISE_MS) / DECAY_TAU_MS);
        glow[slot] = level;
        trail[slot] = Math.max(trail[slot] * trailDecay, level * 0.5);
        if (age > RISE_MS && level < 0.01 && trail[slot] < 0.01) {
          glow[slot] = 0;
          trail[slot] = 0;
          active.delete(slot);
        }
      }
      if (active.size) {
        points.geometry.attributes.glow.needsUpdate = true;
        points.geometry.attributes.trail.needsUpdate = true;
      }

      let shown = 0;
      const P = points.geometry.attributes.position.array;
      for (let i = 0; i < packets.length; i++) {
        const p = packets[i];
        const f = (elapsed - p.start) / p.dur;
        if (f < 0 || f > 1) {
          packetAlpha[i] = 0;
          continue;
        }
        for (let k = 0; k < 3; k++) {
          packetPos[i * 3 + k] = P[p.a * 3 + k] + (P[p.b * 3 + k] - P[p.a * 3 + k]) * f;
        }
        packetAlpha[i] = Math.sin(f * Math.PI) * 0.6;
        shown++;
      }
      for (let i = packets.length; i < MAX_PACKETS; i++) packetAlpha[i] = 0;
      if (packets.length || shown) {
        packetGeometry.attributes.position.needsUpdate = true;
        packetGeometry.attributes.alpha.needsUpdate = true;
      }
    }

    if (shell) {
      shellPulse *= Math.exp(-dt / 500);
      shell.material.opacity = 0.22 + 0.12 * shellPulse;
    }

  }

  function render() {
    controls.update();
    if (container.clientWidth > 0) renderer.render(scene, camera);
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(now - last, 100);
    last = now;
    update(now, dt);
    render();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // Snapshot for tests: how much of the network is visibly lit right now.
  function stats() {
    let lit = 0, outLit = 0;
    if (glow) for (const slot of active) if (glow[slot] > 0.15) lit++;
    for (const slot of outSlots) if (glow[slot] > 0.5) outLit++;
    let brightest = 0;
    if (glow) for (const slot of active) brightest = Math.max(brightest, glow[slot]);
    return { active: active.size, lit, outLit, brightest, packets: packetAlpha.reduce((c, a) => c + (a > 0.05), 0) };
  }

  window.brain3d = { setNodes, play, reveal, setThinking, stats, _step: (now, dt) => { update(now, dt); render(); } };
  if (window.__pendingCircuit) {
    const c = window.__pendingCircuit;
    setNodes(c.nodes, c.colors, c.edges);
  }
}

const container = document.getElementById("brain3d");
if (container) init(container);
