export interface RenderClock {
  now(): number;
  requestFrame(callback: FrameRequestCallback): number;
  cancelFrame(id: number): void;
  setTimer(callback: () => void, delay: number): number;
  clearTimer(id: number): void;
}

const browserClock: RenderClock = {
  now: () => performance.now(),
  requestFrame: callback => requestAnimationFrame(callback),
  cancelFrame: id => cancelAnimationFrame(id),
  setTimer: (callback, delay) => window.setTimeout(callback, delay),
  clearTimer: id => window.clearTimeout(id),
};

/** One coalesced draw per 1/30 second; returning true continues camera damping. */
export function createRenderScheduler(render: () => boolean, clock: RenderClock = browserClock) {
  const interval = 1000 / 30;
  let frame: number | null = null;
  let timer: number | null = null;
  let lastFrame = -Infinity;
  let dirty = false;
  let visible = true;
  let disposed = false;
  let rendering = false;

  const cancel = () => {
    if (frame !== null) clock.cancelFrame(frame);
    if (timer !== null) clock.clearTimer(timer);
    frame = timer = null;
  };
  const schedule = () => {
    if (disposed || !visible || !dirty || rendering || frame !== null || timer !== null) return;
    const remaining = interval - (clock.now() - lastFrame);
    if (remaining > 0) {
      timer = clock.setTimer(() => { timer = null; schedule(); }, remaining);
    } else {
      frame = clock.requestFrame(() => {
        frame = null;
        if (disposed || !visible || !dirty) return;
        dirty = false;
        lastFrame = clock.now();
        rendering = true;
        try { if (render()) dirty = true; }
        finally { rendering = false; schedule(); }
      });
    }
  };
  return {
    invalidate() { if (!disposed) { dirty = true; schedule(); } },
    setVisible(value: boolean) {
      if (disposed || visible === value) return;
      visible = value;
      if (!visible) cancel();
      else { dirty = true; schedule(); }
    },
    dispose() { disposed = true; dirty = false; cancel(); },
  };
}
