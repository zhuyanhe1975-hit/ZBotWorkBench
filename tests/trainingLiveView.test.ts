import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trainingDisplayPosition } from '../src/components/TrainingLiveView';

test('training display applies only the fixed environment layout offset', () => {
  const position = trainingDisplayPosition([13, 1.5, .25], [10, 2, 0], -.82, .82);
  assert.ok(Math.abs(position[0] - 2.18) < 1e-12 && Math.abs(position[1] - .32) < 1e-12 && position[2] === .25);
  assert.deepEqual(trainingDisplayPosition([10, 2, .1], [10, 2, 0], 0, 0), [0, 0, .1]);
});
