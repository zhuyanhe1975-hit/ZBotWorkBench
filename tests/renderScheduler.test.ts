import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRenderScheduler, type RenderClock } from '../src/utils/renderScheduler';

function fakeClock() {
  let time = 0;
  let nextId = 0;
  const frames = new Map<number, FrameRequestCallback>();
  const timers = new Map<number, { at: number; callback: () => void }>();
  const clock: RenderClock = {
    now: () => time,
    requestFrame: callback => { const id = ++nextId; frames.set(id, callback); return id; },
    cancelFrame: id => { frames.delete(id); },
    setTimer: (callback, delay) => { const id = ++nextId; timers.set(id, { at: time + delay, callback }); return id; },
    clearTimer: id => { timers.delete(id); },
  };
  return { clock, frames, timers,
    advance(ms: number) {
      const target = time + ms;
      for (;;) {
        const due = [...timers.entries()].filter(([, t]) => t.at <= target).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        time = due[1].at; timers.delete(due[0]); due[1].callback();
      }
      time = target;
    },
    frame() {
      const pending = [...frames.values()]; frames.clear();
      pending.forEach(callback => callback(time));
    },
  };
}

test('render invalidations coalesce and a settled viewport has no scheduled work', () => {
  const fake = fakeClock(); let renders = 0;
  const scheduler = createRenderScheduler(() => { renders++; return false; }, fake.clock);
  assert.equal(fake.frames.size + fake.timers.size, 0);
  for (let i = 0; i < 30; i++) scheduler.invalidate();
  assert.equal(fake.frames.size, 1);
  fake.frame();
  assert.equal(renders, 1);
  assert.equal(fake.frames.size + fake.timers.size, 0);
  fake.advance(10_000); fake.frame();
  assert.equal(renders, 1);
});

test('active rendering is capped at 30 Hz without spinning RAF between frames', () => {
  const fake = fakeClock(); const timestamps: number[] = [];
  const scheduler = createRenderScheduler(() => { timestamps.push(fake.clock.now()); return timestamps.length < 3; }, fake.clock);
  scheduler.invalidate(); fake.frame();
  assert.equal(fake.frames.size, 0);
  assert.equal(fake.timers.size, 1);
  fake.advance(33); fake.frame();
  assert.equal(timestamps.length, 1);
  fake.advance(1); fake.frame();
  fake.advance(34); fake.frame();
  assert.deepEqual(timestamps, [0, 34, 68]);
  assert.equal(fake.frames.size + fake.timers.size, 0);
});

test('invalidating during rendering is retained as one follow-up frame', () => {
  const fake = fakeClock(); let renders = 0;
  const scheduler = createRenderScheduler(() => {
    if (++renders === 1) { scheduler.invalidate(); scheduler.invalidate(); }
    return false;
  }, fake.clock);
  scheduler.invalidate(); fake.frame();
  assert.equal(fake.frames.size + fake.timers.size, 1);
  fake.advance(34); fake.frame();
  assert.equal(renders, 2);
  assert.equal(fake.frames.size + fake.timers.size, 0);
});

test('invisible viewports cancel RAF and retain pending changes until visible', () => {
  const fake = fakeClock(); let renders = 0;
  const scheduler = createRenderScheduler(() => { renders++; return false; }, fake.clock);
  scheduler.invalidate(); scheduler.setVisible(false);
  scheduler.invalidate(); fake.advance(1000); fake.frame();
  assert.equal(renders, 0);
  assert.equal(fake.frames.size + fake.timers.size, 0);
  scheduler.setVisible(true); fake.frame();
  assert.equal(renders, 1);
  assert.equal(fake.frames.size + fake.timers.size, 0);
});

test('visibility interrupts damping timers and resumes only once', () => {
  const fake = fakeClock(); let renders = 0;
  const scheduler = createRenderScheduler(() => ++renders < 2, fake.clock);
  scheduler.invalidate(); fake.frame(); scheduler.setVisible(false);
  assert.equal(fake.frames.size + fake.timers.size, 0);
  fake.advance(1000); scheduler.setVisible(true); scheduler.setVisible(true); fake.frame();
  assert.equal(renders, 2);
  assert.equal(fake.frames.size + fake.timers.size, 0);
});

test('dispose cancels queued frames and timers and ignores future invalidations', () => {
  for (const afterFrame of [false, true]) {
    const fake = fakeClock(); let renders = 0;
    const scheduler = createRenderScheduler(() => { renders++; return true; }, fake.clock);
    scheduler.invalidate(); if (afterFrame) fake.frame();
    scheduler.dispose(); scheduler.dispose(); scheduler.invalidate(); scheduler.setVisible(true);
    fake.advance(1000); fake.frame();
    assert.equal(renders, afterFrame ? 1 : 0);
    assert.equal(fake.frames.size + fake.timers.size, 0);
  }
});

test('hiding or disposing from within a render cannot schedule a continuation', () => {
  for (const dispose of [false, true]) {
    const fake = fakeClock();
    const scheduler = createRenderScheduler(() => {
      if (dispose) scheduler.dispose(); else scheduler.setVisible(false);
      return true;
    }, fake.clock);
    scheduler.invalidate(); fake.frame();
    assert.equal(fake.frames.size + fake.timers.size, 0);
  }
});
