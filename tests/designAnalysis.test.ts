import test from 'node:test';
import assert from 'node:assert/strict';
import { designConfiguration, jointSample, poseQuality, symmetricEigenvalues } from '../src/utils/designAnalysis';
import { forwardKinematics } from '../src/utils/kinematics';
import { validateConfiguration } from '../src/utils/configuration';

test('designs produce valid six and seven joint configurations and shared samples', () => {
  for (const n of [6,7]) {
    const c = designConfiguration(Array(n-1).fill(180));
    assert.deepEqual(validateConfiguration(c), []);
    assert.equal(c.modules.length, n);
    assert.ok(jointSample(123, n).every(q => q >= -180 && q <= 180));
  }
  assert.deepEqual(jointSample(123, 6), jointSample(123, 7).slice(0,6));
});

test('symmetric eigensolver resolves rotated and rank deficient systems', () => {
  const values = symmetricEigenvalues([[2,1,0],[1,2,0],[0,0,0]]);
  [0,1,3].forEach((v,i) => assert.ok(Math.abs(values[i]-v)<1e-10));
  const planar = poseQuality(forwardKinematics(designConfiguration([0,0,0,0,0]), jointSample(15,6)));
  assert.ok(planar.minimum < 1e-7);
});

test('payload gravity torque agrees with finite difference of tip height', () => {
  const c = designConfiguration([90,180,270,90,180,90]), q = jointSample(88,7);
  const quality = poseQuality(forwardKinematics(c,q));
  for (let i=0;i<7;i++) {
    const a = [...q], b = [...q], delta = 1e-4;
    a[i] += delta * 180 / Math.PI; b[i] -= delta * 180 / Math.PI;
    const expected = -9.81 * (forwardKinematics(c,a).tip[2]-forwardKinematics(c,b).tip[2])/(2*delta);
    assert.ok(Math.abs(expected-quality.payloadTorquePerKg[i])<1e-7);
  }
});
