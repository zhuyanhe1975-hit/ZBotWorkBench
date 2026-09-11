import assert from 'node:assert/strict';
import test from 'node:test';
import { loadCheckpoint, inferPolicy } from '../src/rl/checkpoint';

// torch.save of Linear(3,2) -> ELU -> Linear(2,1), generated using PyTorch CPU.
const fixture = 'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAWBAAAABtb2RlbF9zdGF0ZV9kaWN0cQF9cQIoWA4AAABhY3Rvci4wLndlaWdodHEDY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQQoKFgHAAAAc3RvcmFnZXEFY3RvcmNoCkZsb2F0U3RvcmFnZQpxBlgBAAAAMHEHWAMAAABjcHVxCEsGdHEJUUsASwJLA4ZxCksDSwGGcQuJY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnEMKVJxDXRxDlJxD1gMAAAAYWN0b3IuMC5iaWFzcRBoBCgoaAVoBlgBAAAAMXERaAhLAnRxElFLAEsChXETSwGFcRSJaAwpUnEVdHEWUnEXWA4AAABhY3Rvci4yLndlaWdodHEYaAQoKGgFaAZYAQAAADJxGWgISwJ0cRpRSwBLAUsChnEbSwJLAYZxHIloDClScR10cR5ScR9YDAAAAGFjdG9yLjIuYmlhc3EgaAQoKGgFaAZYAQAAADNxIWgISwF0cSJRSwBLAYVxI0sBhXEkiWgMKVJxJXRxJlJxJ3VzLlBLBwj1vez3oQEAAKEBAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABcAGgBhcmNoaXZlLy5mb3JtYXRfdmVyc2lvbkZCFgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMVBLBwi379yDAQAAAAEAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABoANwBhcmNoaXZlLy5zdG9yYWdlX2FsaWdubWVudEZCMwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlo2NFBLBwg/d3HpAgAAAAIAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABEAPwBhcmNoaXZlL2J5dGVvcmRlckZCOwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWmxpdHRsZVBLBwiFPeMZBgAAAAYAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APgBhcmNoaXZlL2RhdGEvMEZCOgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaAACAPwAAAMAAAAA/AACAvgAAAD8AAABAUEsHCDrZ2IgYAAAAGAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgAsAGFyY2hpdmUvZGF0YS8xRkIoAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAAAC/AACAPlBLBwhMxVcNCAAAAAgAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APABhcmNoaXZlL2RhdGEvMkZCOABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWgAAAEAAAIC/UEsHCHaxjF0IAAAACAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgA8AGFyY2hpdmUvZGF0YS8zRkI4AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaAAAAPlBLBwi3wiXgBAAAAAQAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA8APwBhcmNoaXZlL3ZlcnNpb25GQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4AMgBhcmNoaXZlLy5kYXRhL3NlcmlhbGl6YXRpb25faWRGQi4AWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjEwNzc4NjI3Mjk3OTk4NTQxNjI1MTIwNzQyMDk3MjI4NzkyOTM2MjNQSwcIqCtTWygAAAAoAAAAUEsBAgAAAAAICAAAAAAAAPW97PehAQAAoQEAABAAAAAAAAAAAAAAAAAAAAAAAGFyY2hpdmUvZGF0YS5wa2xQSwECAAAAAAgIAAAAAAAAt+/cgwEAAAABAAAAFwAAAAAAAAAAAAAAAADxAQAAYXJjaGl2ZS8uZm9ybWF0X3ZlcnNpb25QSwECAAAAAAgIAAAAAAAAP3dx6QIAAAACAAAAGgAAAAAAAAAAAAAAAABRAgAAYXJjaGl2ZS8uc3RvcmFnZV9hbGlnbm1lbnRQSwECAAAAAAgIAAAAAAAAhT3jGQYAAAAGAAAAEQAAAAAAAAAAAAAAAADSAgAAYXJjaGl2ZS9ieXRlb3JkZXJQSwECAAAAAAgIAAAAAAAAOtnYiBgAAAAYAAAADgAAAAAAAAAAAAAAAABWAwAAYXJjaGl2ZS9kYXRhLzBQSwECAAAAAAgIAAAAAAAATMVXDQgAAAAIAAAADgAAAAAAAAAAAAAAAADoAwAAYXJjaGl2ZS9kYXRhLzFQSwECAAAAAAgIAAAAAAAAdrGMXQgAAAAIAAAADgAAAAAAAAAAAAAAAABYBAAAYXJjaGl2ZS9kYXRhLzJQSwECAAAAAAgIAAAAAAAAt8Il4AQAAAAEAAAADgAAAAAAAAAAAAAAAADYBAAAYXJjaGl2ZS9kYXRhLzNQSwECAAAAAAgIAAAAAAAA0Z5nVQIAAAACAAAADwAAAAAAAAAAAAAAAABUBQAAYXJjaGl2ZS92ZXJzaW9uUEsBAgAAAAAICAAAAAAAAKgrU1soAAAAKAAAAB4AAAAAAAAAAAAAAAAA0gUAAGFyY2hpdmUvLmRhdGEvc2VyaWFsaXphdGlvbl9pZFBLBgYsAAAAAAAAAB4DLQAAAAAAAAAAAAoAAAAAAAAACgAAAAAAAACDAgAAAAAAAHgGAAAAAAAAUEsGBwAAAAD7CAAAAAAAAAEAAABQSwUGAAAAAAoACgCDAgAAeAYAAAAA';
function bytes(): ArrayBuffer { return Uint8Array.from(Buffer.from(fixture, 'base64')).buffer; }

// Build a minimal stored ZIP so malformed pickle paths are exercised independently.
function archive(pickle: Uint8Array): ArrayBuffer {
  const name = new TextEncoder().encode('data.pkl');
  const buffer = new ArrayBuffer(30 + name.length + pickle.length + 46 + name.length + 22);
  const b = new Uint8Array(buffer), v = new DataView(buffer);
  v.setUint32(0, 0x04034b50, true); v.setUint16(26, name.length, true);
  b.set(name, 30); b.set(pickle, 30 + name.length);
  const c = 30 + name.length + pickle.length;
  v.setUint32(c, 0x02014b50, true); v.setUint32(c + 20, pickle.length, true); v.setUint32(c + 24, pickle.length, true); v.setUint16(c + 28, name.length, true); b.set(name, c + 46);
  const end = c + 46 + name.length;
  v.setUint32(end, 0x06054b50, true); v.setUint16(end + 10, 1, true); v.setUint32(end + 16, c, true);
  return buffer;
}
test('loads real torch.save tensor storage and matches PyTorch CPU ELU output', () => {
  const policy = loadCheckpoint(bytes());
  assert.equal(policy.inputSize, 3); assert.equal(policy.outputSize, 1);
  assert.equal(policy.layers.length, 2);
  assert.deepEqual([...policy.layers[0].weights], [1, -2, .5, -.25, .5, 2]);
  assert.ok(Math.abs(inferPolicy(policy, [.2, .4, -.3])[0] - -1.1207211017608643) < 1e-6);
  assert.equal(loadCheckpoint(bytes(), 'relu').activation, 'relu');
});
test('rejects unsafe pickle globals without executing code', () => {
  const payload = new TextEncoder().encode('cos\nsystem\n.');
  assert.throws(() => loadCheckpoint(archive(payload)), /GLOBAL os.system/);
});
test('rejects corrupt ZIP, truncated pickle, unsupported opcodes and compression', () => {
  for (const input of [new ArrayBuffer(0), bytes().slice(0, 40), archive(new Uint8Array([0x58, 255])), archive(new Uint8Array([0xff]))]) {
    assert.throws(() => loadCheckpoint(input));
  }
  const compressed = archive(new Uint8Array([0x4e, 0x2e]));
  new DataView(compressed).setUint16(40 + 10, 8, true);
  assert.throws(() => loadCheckpoint(compressed), /压缩/);
});
test('inference rejects incompatible and non-finite observations', () => {
  const policy = loadCheckpoint(bytes());
  assert.throws(() => inferPolicy(policy, [0]), /观测维度/);
  assert.throws(() => inferPolicy(policy, [0, NaN, 0]), /NaN/);
});

test('rejects malformed state dictionaries and unsupported external normalizers', () => {
  for (const name of ['student_state_dict', 'actor_state_dict', 'obs_norm_state_dict']) {
    const key = new TextEncoder().encode(name);
    const pickle = new Uint8Array(1 + 5 + key.length + 3);
    pickle[0] = 0x7d; pickle[1] = 0x58;
    new DataView(pickle.buffer).setUint32(2, key.length, true);
    pickle.set(key, 6); pickle.set([0x4e, 0x73, 0x2e], 6 + key.length);
    assert.throws(() => loadCheckpoint(archive(pickle)), /state_dict 无效|外置观测归一化/);
  }
});

test('bounds total pickle allocations even when stack and memo remain small', () => {
  // Each EMPTY_DICT + APPEND pair allocates a Map while stack depth stays <= 2.
  const pickle = new Uint8Array(200_004);
  pickle[0] = 0x5d;
  for (let i = 1; i < pickle.length - 1; i += 2) { pickle[i] = 0x7d; pickle[i + 1] = 0x61; }
  pickle[pickle.length - 1] = 0x2e;
  assert.throws(() => loadCheckpoint(archive(pickle)), /最多 20 万操作/);
});
test('STACK_GLOBAL does not coerce container objects to strings', () => {
  assert.throws(() => loadCheckpoint(archive(new Uint8Array([0x5d, 0x5d, 0x93, 0x2e]))), /名称必须是字符串/);
});

// Modern actor/student archives produced with torch.save and RSL-RL EmpiricalNormalization.
const modernFixtures = {"actor_state_dict":{"base64":"UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAWBAAAABhY3Rvcl9zdGF0ZV9kaWN0cQF9cQIoWAwAAABtbHAuMC53ZWlnaHRxA2N0b3JjaC5fdXRpbHMKX3JlYnVpbGRfdGVuc29yX3YyCnEEKChYBwAAAHN0b3JhZ2VxBWN0b3JjaApGbG9hdFN0b3JhZ2UKcQZYAQAAADBxB1gDAAAAY3B1cQhLBnRxCVFLAEsCSwOGcQpLA0sBhnELiWNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxDClScQ10cQ5ScQ9YCgAAAG1scC4wLmJpYXNxEGgEKChoBWgGWAEAAAAxcRFoCEsCdHESUUsASwKFcRNLAYVxFIloDClScRV0cRZScRdYDAAAAG1scC4yLndlaWdodHEYaAQoKGgFaAZYAQAAADJxGWgISwJ0cRpRSwBLAUsChnEbSwJLAYZxHIloDClScR10cR5ScR9YCgAAAG1scC4yLmJpYXNxIGgEKChoBWgGWAEAAAAzcSFoCEsBdHEiUUsASwGFcSNLAYVxJIloDClScSV0cSZScSdYFgAAAGRpc3RyaWJ1dGlvbi5zdGRfcGFyYW1xKGgEKChoBWgGWAEAAAA0cSloCEsBdHEqUUsASwGFcStLAYVxLIloDClScS10cS5ScS9YFAAAAG9ic19ub3JtYWxpemVyLl9tZWFucTBoBCgoaAVoBlgBAAAANXExaAhLA3RxMlFLAEsBSwOGcTNLA0sBhnE0iWgMKVJxNXRxNlJxN1gTAAAAb2JzX25vcm1hbGl6ZXIuX3ZhcnE4aAQoKGgFaAZYAQAAADZxOWgISwN0cTpRSwBLAUsDhnE7SwNLAYZxPIloDClScT10cT5ScT9YEwAAAG9ic19ub3JtYWxpemVyLl9zdGRxQGgEKChoBWgGWAEAAAA3cUFoCEsDdHFCUUsASwFLA4ZxQ0sDSwGGcUSJaAwpUnFFdHFGUnFHWBQAAABvYnNfbm9ybWFsaXplci5jb3VudHFIaAQoKGgFY3RvcmNoCkxvbmdTdG9yYWdlCnFJWAEAAAA4cUpoCEsBdHFLUUsAKSmJaAwpUnFMdHFNUnFOdXMuUEsHCF3UL7AsAwAALAMAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAAFwAPAGFyY2hpdmUvLmZvcm1hdF92ZXJzaW9uRkILAFpaWlpaWlpaWlpaMVBLBwi379yDAQAAAAEAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABoANwBhcmNoaXZlLy5zdG9yYWdlX2FsaWdubWVudEZCMwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlo2NFBLBwg/d3HpAgAAAAIAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABEAPwBhcmNoaXZlL2J5dGVvcmRlckZCOwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWmxpdHRsZVBLBwiFPeMZBgAAAAYAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APgBhcmNoaXZlL2RhdGEvMEZCOgBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaAACAPwAAAMAAAAA/AACAvgAAAD8AAABAUEsHCDrZ2IgYAAAAGAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgAsAGFyY2hpdmUvZGF0YS8xRkIoAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAAAC/AACAPlBLBwhMxVcNCAAAAAgAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APABhcmNoaXZlL2RhdGEvMkZCOABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWgAAAEAAAIC/UEsHCHaxjF0IAAAACAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgA8AGFyY2hpdmUvZGF0YS8zRkI4AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaAAAAPlBLBwi3wiXgBAAAAAQAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4AQABhcmNoaXZlL2RhdGEvNEZCPABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlrNzMw9UEsHCLBS8QIEAAAABAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgBAAGFyY2hpdmUvZGF0YS81RkI8AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWs3MzD3NzEy+mpmZPlBLBwiwHT7fDAAAAAwAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4AOABhcmNoaXZlL2RhdGEvNkZCNABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaAAAAAAAAgD4AAIBAUEsHCIsNKzYMAAAADAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgA4AGFyY2hpdmUvZGF0YS83RkI0AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAAAAAAAAAPwAAAEBQSwcIqKh4LgwAAAAMAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOADgAYXJjaGl2ZS9kYXRhLzhGQjQAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWnsAAAAAAAAAUEsHCK9SGJQIAAAACAAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADwA7AGFyY2hpdmUvdmVyc2lvbkZCNwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMwpQSwcI0Z5nVQIAAAACAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAeADIAYXJjaGl2ZS8uZGF0YS9zZXJpYWxpemF0aW9uX2lkRkIuAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlowMTk3Nzg2MDM1MzM2OTIwMTQ0MzEzMTY3NDIxNDUxNzUxOTA0MzMzUEsHCCo4JVYoAAAAKAAAAFBLAQIAAAAACAgAAAAAAABd1C+wLAMAACwDAAAQAAAAAAAAAAAAAAAAAAAAAABhcmNoaXZlL2RhdGEucGtsUEsBAgAAAAAICAAAAAAAALfv3IMBAAAAAQAAABcAAAAAAAAAAAAAAAAAfAMAAGFyY2hpdmUvLmZvcm1hdF92ZXJzaW9uUEsBAgAAAAAICAAAAAAAAD93cekCAAAAAgAAABoAAAAAAAAAAAAAAAAA0QMAAGFyY2hpdmUvLnN0b3JhZ2VfYWxpZ25tZW50UEsBAgAAAAAICAAAAAAAAIU94xkGAAAABgAAABEAAAAAAAAAAAAAAAAAUgQAAGFyY2hpdmUvYnl0ZW9yZGVyUEsBAgAAAAAICAAAAAAAADrZ2IgYAAAAGAAAAA4AAAAAAAAAAAAAAAAA1gQAAGFyY2hpdmUvZGF0YS8wUEsBAgAAAAAICAAAAAAAAEzFVw0IAAAACAAAAA4AAAAAAAAAAAAAAAAAaAUAAGFyY2hpdmUvZGF0YS8xUEsBAgAAAAAICAAAAAAAAHaxjF0IAAAACAAAAA4AAAAAAAAAAAAAAAAA2AUAAGFyY2hpdmUvZGF0YS8yUEsBAgAAAAAICAAAAAAAALfCJeAEAAAABAAAAA4AAAAAAAAAAAAAAAAAWAYAAGFyY2hpdmUvZGF0YS8zUEsBAgAAAAAICAAAAAAAALBS8QIEAAAABAAAAA4AAAAAAAAAAAAAAAAA1AYAAGFyY2hpdmUvZGF0YS80UEsBAgAAAAAICAAAAAAAALAdPt8MAAAADAAAAA4AAAAAAAAAAAAAAAAAVAcAAGFyY2hpdmUvZGF0YS81UEsBAgAAAAAICAAAAAAAAIsNKzYMAAAADAAAAA4AAAAAAAAAAAAAAAAA3AcAAGFyY2hpdmUvZGF0YS82UEsBAgAAAAAICAAAAAAAAKioeC4MAAAADAAAAA4AAAAAAAAAAAAAAAAAXAgAAGFyY2hpdmUvZGF0YS83UEsBAgAAAAAICAAAAAAAAK9SGJQIAAAACAAAAA4AAAAAAAAAAAAAAAAA3AgAAGFyY2hpdmUvZGF0YS84UEsBAgAAAAAICAAAAAAAANGeZ1UCAAAAAgAAAA8AAAAAAAAAAAAAAAAAWAkAAGFyY2hpdmUvdmVyc2lvblBLAQIAAAAACAgAAAAAAAAqOCVWKAAAACgAAAAeAAAAAAAAAAAAAAAAANIJAABhcmNoaXZlLy5kYXRhL3NlcmlhbGl6YXRpb25faWRQSwYGLAAAAAAAAAAeAy0AAAAAAAAAAAAPAAAAAAAAAA8AAAAAAAAArwMAAAAAAAB4CgAAAAAAAFBLBgcAAAAAJw4AAAAAAAABAAAAUEsFBgAAAAAPAA8ArwMAAHgKAAAAAA==","output":[15.016131401062012]},"student_state_dict":{"base64":"UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAWBIAAABzdHVkZW50X3N0YXRlX2RpY3RxAX1xAihYDAAAAG1scC4wLndlaWdodHEDY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQQoKFgHAAAAc3RvcmFnZXEFY3RvcmNoCkZsb2F0U3RvcmFnZQpxBlgBAAAAMHEHWAMAAABjcHVxCEsGdHEJUUsASwJLA4ZxCksDSwGGcQuJY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnEMKVJxDXRxDlJxD1gKAAAAbWxwLjAuYmlhc3EQaAQoKGgFaAZYAQAAADFxEWgISwJ0cRJRSwBLAoVxE0sBhXEUiWgMKVJxFXRxFlJxF1gMAAAAbWxwLjIud2VpZ2h0cRhoBCgoaAVoBlgBAAAAMnEZaAhLAnRxGlFLAEsBSwKGcRtLAksBhnEciWgMKVJxHXRxHlJxH1gKAAAAbWxwLjIuYmlhc3EgaAQoKGgFaAZYAQAAADNxIWgISwF0cSJRSwBLAYVxI0sBhXEkiWgMKVJxJXRxJlJxJ1gWAAAAZGlzdHJpYnV0aW9uLnN0ZF9wYXJhbXEoaAQoKGgFaAZYAQAAADRxKWgISwF0cSpRSwBLAYVxK0sBhXEsiWgMKVJxLXRxLlJxL3VzLlBLBwjeLgZ+6QEAAOkBAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAABcAEgBhcmNoaXZlLy5mb3JtYXRfdmVyc2lvbkZCDgBaWlpaWlpaWlpaWlpaWjFQSwcIt+/cgwEAAAABAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAaADcAYXJjaGl2ZS8uc3RvcmFnZV9hbGlnbm1lbnRGQjMAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaNjRQSwcIP3dx6QIAAAACAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAARAD8AYXJjaGl2ZS9ieXRlb3JkZXJGQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpsaXR0bGVQSwcIhT3jGQYAAAAGAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOAD4AYXJjaGl2ZS9kYXRhLzBGQjoAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWgAAgD8AAADAAAAAPwAAgL4AAAA/AAAAQFBLBwg62diIGAAAABgAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4ALABhcmNoaXZlL2RhdGEvMUZCKABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaAAAAvwAAgD5QSwcITMVXDQgAAAAIAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOADwAYXJjaGl2ZS9kYXRhLzJGQjgAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAAABAAACAv1BLBwh2sYxdCAAAAAgAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA4APABhcmNoaXZlL2RhdGEvM0ZCOABaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWgAAAD5QSwcIt8Il4AQAAAAEAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOAEAAYXJjaGl2ZS9kYXRhLzRGQjwAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpazczMPVBLBwiwUvECBAAAAAQAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA8APwBhcmNoaXZlL3ZlcnNpb25GQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAB4AMgBhcmNoaXZlLy5kYXRhL3NlcmlhbGl6YXRpb25faWRGQi4AWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWjAwNTUxOTAxODUzMjAzNTkwNzg3MDgxOTc3MjU2NDk3MzIwNzUyNjJQSwcIXdHt8CgAAAAoAAAAUEsBAgAAAAAICAAAAAAAAN4uBn7pAQAA6QEAABAAAAAAAAAAAAAAAAAAAAAAAGFyY2hpdmUvZGF0YS5wa2xQSwECAAAAAAgIAAAAAAAAt+/cgwEAAAABAAAAFwAAAAAAAAAAAAAAAAA5AgAAYXJjaGl2ZS8uZm9ybWF0X3ZlcnNpb25QSwECAAAAAAgIAAAAAAAAP3dx6QIAAAACAAAAGgAAAAAAAAAAAAAAAACRAgAAYXJjaGl2ZS8uc3RvcmFnZV9hbGlnbm1lbnRQSwECAAAAAAgIAAAAAAAAhT3jGQYAAAAGAAAAEQAAAAAAAAAAAAAAAAASAwAAYXJjaGl2ZS9ieXRlb3JkZXJQSwECAAAAAAgIAAAAAAAAOtnYiBgAAAAYAAAADgAAAAAAAAAAAAAAAACWAwAAYXJjaGl2ZS9kYXRhLzBQSwECAAAAAAgIAAAAAAAATMVXDQgAAAAIAAAADgAAAAAAAAAAAAAAAAAoBAAAYXJjaGl2ZS9kYXRhLzFQSwECAAAAAAgIAAAAAAAAdrGMXQgAAAAIAAAADgAAAAAAAAAAAAAAAACYBAAAYXJjaGl2ZS9kYXRhLzJQSwECAAAAAAgIAAAAAAAAt8Il4AQAAAAEAAAADgAAAAAAAAAAAAAAAAAYBQAAYXJjaGl2ZS9kYXRhLzNQSwECAAAAAAgIAAAAAAAAsFLxAgQAAAAEAAAADgAAAAAAAAAAAAAAAACUBQAAYXJjaGl2ZS9kYXRhLzRQSwECAAAAAAgIAAAAAAAA0Z5nVQIAAAACAAAADwAAAAAAAAAAAAAAAAAUBgAAYXJjaGl2ZS92ZXJzaW9uUEsBAgAAAAAICAAAAAAAAF3R7fAoAAAAKAAAAB4AAAAAAAAAAAAAAAAAkgYAAGFyY2hpdmUvLmRhdGEvc2VyaWFsaXphdGlvbl9pZFBLBgYsAAAAAAAAAB4DLQAAAAAAAAAAAAsAAAAAAAAACwAAAAAAAAC/AgAAAAAAADgHAAAAAAAAUEsGBwAAAAD3CQAAAAAAAAEAAABQSwUGAAAAAAsACwC/AgAAOAcAAAAA","output":[-1.1207211017608643]}};

function modernBytes(kind: keyof typeof modernFixtures = 'actor_state_dict'): ArrayBuffer {
  return Uint8Array.from(Buffer.from(modernFixtures[kind].base64, 'base64')).buffer;
}
function entry(buffer: ArrayBuffer, name: string): Uint8Array {
  const b = new Uint8Array(buffer), v = new DataView(buffer);
  // Test-fixture inspection only; locate central records and their referenced data.
  for (let c = 0; c + 46 <= b.length; c++) {
    if (v.getUint32(c, true) !== 0x02014b50) continue;
    const n = v.getUint16(c + 28, true);
    if (new TextDecoder().decode(b.subarray(c + 46, c + 46 + n)) !== `archive/${name}`) continue;
    const local = v.getUint32(c + 42, true);
    const start = local + 30 + v.getUint16(local + 26, true) + v.getUint16(local + 28, true);
    return b.subarray(start, start + v.getUint32(c + 24, true));
  }
  throw new Error(`Missing fixture entry ${name}`);
}
test('loads modern actor normalizer and student MLP against PyTorch CPU fixtures', () => {
  for (const kind of ['actor_state_dict', 'student_state_dict'] as const) {
    const policy = loadCheckpoint(modernBytes(kind));
    assert.ok(Math.abs(inferPolicy(policy, [.2, .4, -.3])[0] - modernFixtures[kind].output[0]) < 2e-6);
    assert.equal(!!policy.normalization, kind === 'actor_state_dict');
    if (policy.normalization) {
      assert.equal(policy.normalization.epsilon, .01);
      assert.deepEqual([...policy.normalization.std], [0, .5, 2]);
      // Zero variance remains valid because epsilon is added to std, not variance.
      assert.ok(Number.isFinite(inferPolicy(policy, [1, -20, 100])[0]));
    }
  }
});
test('rejects nonfinite, negative and inconsistent normalization statistics', () => {
  for (const [storage, value] of [[5, NaN], [7, Infinity], [7, -1], [6, -1], [7, 1]]) {
    const buffer = modernBytes(), target = entry(buffer, `data/${storage}`);
    new DataView(target.buffer, target.byteOffset, target.byteLength).setFloat32(0, value, true);
    assert.throws(() => loadCheckpoint(buffer), /NaN|Infinity|方差或标准差/);
  }
});
test('rejects malformed normalizer dimensions, missing fields and recurrent layers', () => {
  const buffer = modernBytes(), metadata = entry(buffer, 'data.pkl');
  const name = new TextEncoder().encode('obs_normalizer._mean');
  const start = Buffer.from(metadata).indexOf(name);
  assert.ok(start >= 0);
  let shape = -1;
  for (let i = start + name.length; i < metadata.length - 4; i++) {
    if (metadata[i] === 0x4b && metadata[i + 1] === 1 && metadata[i + 2] === 0x4b && metadata[i + 3] === 3 && metadata[i + 4] === 0x86) { shape = i; break; }
  }
  assert.ok(shape >= 0); metadata[shape + 3] = 2;
  assert.throws(() => loadCheckpoint(buffer), /归一化统计量维度/);
  for (const [from, to, expected] of [
    ['obs_normalizer._std', 'obs_normalizer._bad', /额外层/],
    ['obs_normalizer._std', 'obs_normalizer._var', /统计量不完整/],
    ['mlp.0.weight', 'rnn.0.weight', /循环网络/],
  ] as const) {
    const modified = modernBytes(), meta = entry(modified, 'data.pkl');
    const offset = Buffer.from(meta).indexOf(from);
    assert.ok(offset >= 0); meta.set(new TextEncoder().encode(to), offset);
    assert.throws(() => loadCheckpoint(modified), expected);
  }
});
