/** Restricted data-only reader for torch.save ZIP state dictionaries. Never executes pickle code. */
export type Activation = 'elu' | 'relu' | 'tanh';
export interface PolicyLayer { inputSize: number; outputSize: number; weights: Float32Array; bias: Float32Array }
export interface PolicyNormalization { mean: Float32Array; std: Float32Array; epsilon: number }
export interface PolicyNetwork { normalization?: PolicyNormalization; layers: PolicyLayer[]; activation: Activation; inputSize: number; outputSize: number }
const MAX_FILE = 128 * 1024 * 1024;
const MAX_VALUES = 8_000_000;
const decoder = new TextDecoder('utf-8', { fatal: true });
function fail(message: string): never { throw new Error(`策略文件：${message}`); }
function integer(value: unknown, max = MAX_VALUES): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value > max) fail('无效或过大的张量尺寸');
  return value as number;
}
function zipEntries(buffer: ArrayBuffer): Map<string, Uint8Array> {
  if (buffer.byteLength > MAX_FILE) fail('文件超过 128 MiB 限制');
  const bytes = new Uint8Array(buffer), view = new DataView(buffer);
  let end = bytes.length - 22;
  while (end >= Math.max(0, bytes.length - 65557) && view.getUint32(end, true) !== 0x06054b50) end--;
  if (end < 0 || bytes.length < 22 || view.getUint32(end, true) !== 0x06054b50) fail('需要现代 PyTorch ZIP 格式 .pt/.pth');
  if (view.getUint16(end + 4, true) || view.getUint16(end + 6, true)) fail('不支持分卷 ZIP');
  const count = view.getUint16(end + 10, true);
  let offset = view.getUint32(end + 16, true);
  const entries = new Map<string, Uint8Array>();
  for (let i = 0; i < count; i++) {
    if (offset + 46 > end || view.getUint32(offset, true) !== 0x02014b50) fail('ZIP 目录损坏');
    const flags = view.getUint16(offset + 8, true), compression = view.getUint16(offset + 10, true);
    const size = view.getUint32(offset + 24, true), compressedSize = view.getUint32(offset + 20, true);
    const nameSize = view.getUint16(offset + 28, true), extra = view.getUint16(offset + 30, true), comment = view.getUint16(offset + 32, true);
    const local = view.getUint32(offset + 42, true);
    if (compression !== 0 || flags & 1) fail('不支持压缩或加密 ZIP；请使用 torch.save 默认格式');
    if (size !== compressedSize || offset + 46 + nameSize + extra + comment > end || local + 30 > end) fail('ZIP 数据损坏');
    const name = decoder.decode(bytes.subarray(offset + 46, offset + 46 + nameSize));
    if (entries.has(name)) fail('ZIP 条目重名');
    if (view.getUint32(local, true) !== 0x04034b50) fail('ZIP 本地头损坏');
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    if (start + size > end) fail('ZIP 条目超出文件边界');
    entries.set(name, bytes.subarray(start, start + size));
    offset += 46 + nameSize + extra + comment;
  }
  return entries;
}
interface GlobalRef { global: string }
interface Storage { storage: string; dtype: string; count: number }
interface Tensor { tensor: true; storage: Storage; offset: number; shape: number[]; stride: number[] }
const allowedGlobals = new Set(['collections.OrderedDict', 'torch._utils._rebuild_tensor_v2', 'torch._utils._rebuild_tensor', 'torch.FloatStorage', 'torch.LongStorage']);
function parsePickle(bytes: Uint8Array): unknown {
  if (bytes.length > 8 * 1024 * 1024) fail('pickle 元数据过大');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const stack: any[] = [], memo = new Map<number, any>(), mark = Symbol('mark');
  let position = 0, operations = 0;
  const take = (n: number) => { if (position + n > bytes.length) fail('pickle 数据截断'); const p = position; position += n; return p; };
  const u8 = () => bytes[take(1)];
  const u32 = () => view.getUint32(take(4), true);
  const pop = () => { if (!stack.length) fail('pickle 栈无效'); return stack.pop(); };
  const items = () => { const i = stack.lastIndexOf(mark); if (i < 0) fail('pickle MARK 缺失'); const values = stack.splice(i); return values.slice(1); };
  const line = () => { const start = position; while (u8() !== 10) { /* bounded by take */ } return decoder.decode(bytes.subarray(start, position - 1)); };
  const assign = (values: any[]) => { const target = stack.at(-1); if (!(target instanceof Map) || values.length % 2) fail('pickle 字典无效'); for (let i = 0; i < values.length; i += 2) target.set(values[i], values[i + 1]); };
  const append = (values: any[]) => { const target = stack.at(-1); if (!Array.isArray(target)) fail('pickle 列表无效'); for (const value of values) target.push(value); };
  while (position < bytes.length) {
    // Bound total work/allocations, including containers removed from the stack.
    if (++operations > 200_000 || stack.length > 100_000 || memo.size > 100_000) fail('pickle 结构过大（最多 20 万操作）');
    const opcode = u8();
    switch (opcode) {
      case 0x80: if (u8() > 5) fail('不支持的 pickle 版本'); break;
      case 0x95: { const lo = u32(), hi = u32(); if (hi || position + lo > bytes.length) fail('pickle FRAME 无效'); break; }
      case 0x2e: if (stack.length !== 1) fail('pickle 结束栈无效'); return pop();
      case 0x28: stack.push(mark); break;
      case 0x4e: stack.push(null); break;
      case 0x88: stack.push(true); break;
      case 0x89: stack.push(false); break;
      case 0x4b: stack.push(u8()); break;
      case 0x4d: stack.push(view.getUint16(take(2), true)); break;
      case 0x4a: stack.push(view.getInt32(take(4), true)); break;
      case 0x47: stack.push(view.getFloat64(take(8), false)); break;
      case 0x58: { const n = u32(); const p = take(n); stack.push(decoder.decode(bytes.subarray(p, p + n))); break; }
      case 0x8c: { const n = u8(); const p = take(n); stack.push(decoder.decode(bytes.subarray(p, p + n))); break; }
      case 0x7d: stack.push(new Map()); break;
      case 0x5d: case 0x29: stack.push([]); break;
      case 0x74: stack.push(items()); break;
      case 0x85: stack.push([pop()]); break;
      case 0x86: { const b = pop(), a = pop(); stack.push([a, b]); break; }
      case 0x87: { const c = pop(), b = pop(), a = pop(); stack.push([a, b, c]); break; }
      case 0x75: assign(items()); break;
      case 0x73: { const value = pop(), key = pop(); assign([key, value]); break; }
      case 0x65: append(items()); break;
      case 0x61: { const value = pop(); append([value]); break; }
      case 0x71: memo.set(u8(), stack.at(-1)); break;
      case 0x72: memo.set(u32(), stack.at(-1)); break;
      case 0x94: memo.set(memo.size, stack.at(-1)); break;
      case 0x68: case 0x6a: { const key = opcode === 0x68 ? u8() : u32(); if (!memo.has(key)) fail('pickle 引用无效'); stack.push(memo.get(key)); break; }
      case 0x63: case 0x93: {
        let name: string;
        if (opcode === 0x63) name = `${line()}.${line()}`;
        else {
          const member = pop(), module = pop();
          if (typeof member !== 'string' || typeof module !== 'string') fail('pickle STACK_GLOBAL 名称必须是字符串');
          name = `${module}.${member}`;
        }
        if (!allowedGlobals.has(name)) fail(`不支持的 pickle GLOBAL ${name}（不会执行 Python 代码）`);
        stack.push({ global: name }); break;
      }
      case 0x51: {
        const id = pop();
        if (!Array.isArray(id) || id[0] !== 'storage' || !['torch.FloatStorage', 'torch.LongStorage'].includes(id[1]?.global) || typeof id[2] !== 'string') fail('不支持的存储类型');
        stack.push({ storage: id[2], dtype: id[1].global, count: integer(id[4], MAX_FILE / 4) }); break;
      }
      case 0x52: {
        const args = pop(), fn: GlobalRef = pop();
        if (!Array.isArray(args)) fail('pickle REDUCE 参数无效');
        if (fn?.global === 'collections.OrderedDict' && !args.length) stack.push(new Map());
        else if (fn?.global === 'torch._utils._rebuild_tensor_v2' || fn?.global === 'torch._utils._rebuild_tensor') {
          if (!args[0]?.storage || !Array.isArray(args[2]) || !Array.isArray(args[3]) || args[2].length !== args[3].length || args[2].length > 8) fail('张量描述无效');
          stack.push({ tensor: true, storage: args[0], offset: integer(args[1]), shape: args[2].map(v => integer(v)), stride: args[3].map(v => integer(v)) });
        } else fail('禁止的 pickle REDUCE');
        break;
      }
      case 0x62: { const state = pop(); if (!(stack.at(-1) instanceof Map) || !(state instanceof Map)) fail('禁止的 pickle BUILD'); break; }
      default: fail(`不支持的 pickle 操作 0x${opcode.toString(16)}`);
    }
  }
  return fail('pickle 缺少 STOP');
}
function readTensor(value: any, entries: Map<string, Uint8Array>, prefix: string): { shape: number[]; data: Float32Array } {
  const tensor = value as Tensor;
  if (!tensor?.tensor || tensor.storage.dtype !== 'torch.FloatStorage') fail('策略权重必须是 Float32 张量');
  const bytes = entries.get(`${prefix}data/${tensor.storage.storage}`);
  if (!bytes || bytes.length !== tensor.storage.count * 4) fail('张量存储缺失或尺寸不符');
  const count = tensor.shape.reduce((a, b) => a * b, 1);
  integer(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), data = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let index = tensor.offset, remaining = i;
    for (let d = tensor.shape.length - 1; d >= 0; d--) { index += remaining % tensor.shape[d] * tensor.stride[d]; remaining = Math.floor(remaining / tensor.shape[d]); }
    if (index >= tensor.storage.count) fail('张量索引越界');
    const number = view.getFloat32(index * 4, true);
    if (!Number.isFinite(number)) fail('权重包含 NaN 或 Infinity');
    data[i] = number;
  }
  return { shape: tensor.shape, data };
}
export function loadCheckpoint(buffer: ArrayBuffer, activation: Activation = 'elu'): PolicyNetwork {
  if (!['elu', 'relu', 'tanh'].includes(activation)) fail('不支持的激活函数');
  const entries = zipEntries(buffer), metadata = [...entries.keys()].filter(name => name.endsWith('/data.pkl') || name === 'data.pkl');
  if (metadata.length !== 1) fail('未找到唯一的 data.pkl');
  const prefix = metadata[0].slice(0, -8), byteorder = entries.get(`${prefix}byteorder`);
  if (byteorder && decoder.decode(byteorder) !== 'little') fail('仅支持小端 Float32 权重');
  const root = parsePickle(entries.get(metadata[0])!);
  if (!(root instanceof Map)) fail('需要 state_dict 检查点');
  const containers = ['model_state_dict', 'actor_state_dict', 'student_state_dict'].filter(key => root.has(key));
  if (containers.length > 1) fail('检查点含多个候选策略，无法确定推理网络');
  const modern = containers[0] === 'actor_state_dict' || containers[0] === 'student_state_dict';
  const state = containers.length ? root.get(containers[0]) : root;
  if (!(state instanceof Map)) fail('策略 state_dict 无效');
  if ([...state.keys()].some(k => typeof k === 'string' && /rnn|lstm|gru|memory/i.test(k))) fail('暂不支持循环网络');
  if ([...root.keys()].some(k => typeof k === 'string' && /norm/i.test(k))) fail('暂不支持外置观测归一化检查点');
  if (!modern && [...state.keys()].some(k => typeof k === 'string' && /norm|student/i.test(k))) fail('暂不支持此格式的归一化或学生网络');
  const layerPrefix = modern ? 'mlp' : 'actor';
  const pattern = new RegExp(`^${layerPrefix}\\.\\d+\\.weight$`);
  const weights = [...state.keys()].filter((k): k is string => typeof k === 'string' && pattern.test(k)).sort((a, b) => Number(a.split('.')[1]) - Number(b.split('.')[1]));
  if (!weights.length || weights.length > 16) fail('未找到支持的前馈网络');
  if (weights.some((key, i) => key !== `${layerPrefix}.${i * 2}.weight`)) fail('仅支持 Linear 与单个激活层交替的网络');
  const normalizerKeys = ['obs_normalizer._mean', 'obs_normalizer._std', 'obs_normalizer._var', 'obs_normalizer.count'];
  const expected = new Set(weights.flatMap(k => [k, k.replace(/weight$/, 'bias')]));
  if ([...state.keys()].some(k => typeof k !== 'string' || (!expected.has(k) && (modern
    ? !normalizerKeys.includes(k) && k !== 'distribution.std_param'
    : k.startsWith('actor.'))))) fail('策略含未支持的额外层');
  let parameterCount = 0;
  const layers = weights.map(key => {
    for (const name of [key, key.replace(/weight$/, 'bias')]) {
      const tensor = state.get(name) as Tensor;
      if (!tensor?.tensor) fail('策略张量缺失');
      parameterCount += tensor.shape.reduce((a, b) => a * b, 1);
      if (parameterCount > MAX_VALUES) fail('策略参数总量超过 800 万限制');
    }
    const w = readTensor(state.get(key), entries, prefix), b = readTensor(state.get(key.replace(/weight$/, 'bias')), entries, prefix);
    if (w.shape.length !== 2 || b.shape.length !== 1 || w.shape[0] !== b.shape[0] || !w.shape[0] || !w.shape[1]) fail('Linear 权重与偏置尺寸不符');
    return { inputSize: w.shape[1], outputSize: w.shape[0], weights: w.data, bias: b.data };
  });
  for (let i = 1; i < layers.length; i++) if (layers[i].inputSize !== layers[i - 1].outputSize) fail('网络层尺寸不匹配');
  const inputSize = layers[0].inputSize;
  let normalization: PolicyNormalization | undefined;
  if (modern && normalizerKeys.some(key => state.has(key))) {
    if (parameterCount + inputSize * 3 > MAX_VALUES) fail('含归一化统计量的策略参数总量超过 800 万限制');
    if (!normalizerKeys.every(key => state.has(key))) fail('观测归一化统计量不完整');
    const stats = normalizerKeys.slice(0, 3).map(key => {
      const tensor = state.get(key) as Tensor;
      if (!tensor?.tensor || tensor.shape.length !== 2 || tensor.shape[0] !== 1 || tensor.shape[1] !== inputSize) fail('归一化统计量维度不匹配');
      return readTensor(tensor, entries, prefix).data;
    });
    const [mean, std, variance] = stats;
    if (std.some(v => v < 0) || variance.some(v => v < 0) || std.some((v, i) => Math.abs(v - Math.sqrt(variance[i])) > 1e-5 * Math.max(1, v))) fail('归一化方差或标准差无效');
    const count = state.get('obs_normalizer.count') as Tensor;
    const bytes = count?.tensor && entries.get(`${prefix}data/${count.storage.storage}`);
    if (!count?.tensor || count.shape.length !== 0 || count.storage.dtype !== 'torch.LongStorage' || !bytes || bytes.length !== count.storage.count * 8 || count.offset >= count.storage.count || new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigInt64(count.offset * 8, true) < 0n) fail('归一化样本计数无效');
    // RSL-RL EmpiricalNormalization default: (x - mean) / (std + 1e-2), no clipping.
    normalization = { mean, std, epsilon: 1e-2 };
  }
  return { layers, activation, inputSize, outputSize: layers.at(-1)!.outputSize, ...(normalization ? { normalization } : {}) };
}
export function inferPolicy(policy: PolicyNetwork, observations: ArrayLike<number>): Float32Array {
  if (observations.length !== policy.inputSize) fail(`观测维度应为 ${policy.inputSize}，实际为 ${observations.length}`);
  let input = Float32Array.from(observations);
  if (!input.every(Number.isFinite)) fail('观测包含 NaN 或 Infinity');
  if (policy.normalization) {
    const { mean, std, epsilon } = policy.normalization;
    // Round the intermediate subtraction/addition exactly as PyTorch Float32 does.
    input = input.map((value, i) => Math.fround(value - mean[i]) / Math.fround(std[i] + epsilon));
    if (!input.every(Number.isFinite)) fail('归一化观测包含非有限数值');
  }
  for (let l = 0; l < policy.layers.length; l++) {
    const layer = policy.layers[l], output = new Float32Array(layer.outputSize);
    for (let row = 0; row < layer.outputSize; row++) {
      let sum = layer.bias[row];
      for (let col = 0; col < layer.inputSize; col++) sum += layer.weights[row * layer.inputSize + col] * input[col];
      if (l < policy.layers.length - 1) sum = policy.activation === 'elu' ? (sum >= 0 ? sum : Math.expm1(sum)) : policy.activation === 'relu' ? Math.max(0, sum) : Math.tanh(sum);
      output[row] = sum;
    }
    if (!output.every(Number.isFinite)) fail('推理产生非有限动作');
    input = output;
  }
  return input;
}
