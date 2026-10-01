import { tetrahedronObj } from './tetrahedronGeometry';
import { connectorName } from './moduleMount';
import { ZbotConfiguration } from '../types/zbot';
import { generateMujocoXML, XmlGeneratorOptions } from './xmlGenerator';
import { generateUrdf } from './urdfGenerator';

// Store-only ZIP, using the standard ZIP headers and CRC32; no runtime dependency.
export function createZip(files: Record<string, string>): Uint8Array {
  const encoder = new TextEncoder();
  const local: Uint8Array[] = [], central: Uint8Array[] = [];
  let offset = 0, centralSize = 0;
  for (const [filename, text] of Object.entries(files)) {
    const name = encoder.encode(filename), data = encoder.encode(text);
    let crc = 0xffffffff;
    for (const byte of data) {
      crc ^= byte;
      for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
    crc = (crc ^ 0xffffffff) >>> 0;
    const header = new Uint8Array(30 + name.length), view = new DataView(header.buffer);
    view.setUint32(0, 0x04034b50, true); view.setUint16(4, 20, true); view.setUint16(6, 0x800, true);
    view.setUint16(12, 33, true); view.setUint32(14, crc, true);
    view.setUint32(18, data.length, true); view.setUint32(22, data.length, true); view.setUint16(26, name.length, true);
    header.set(name, 30);
    const entry = new Uint8Array(46 + name.length), cv = new DataView(entry.buffer);
    cv.setUint32(0, 0x02014b50, true); cv.setUint16(4, 20, true); cv.setUint16(6, 20, true); cv.setUint16(8, 0x800, true);
    cv.setUint16(14, 33, true); cv.setUint32(16, crc, true); cv.setUint32(20, data.length, true); cv.setUint32(24, data.length, true);
    cv.setUint16(28, name.length, true); cv.setUint32(42, offset, true); entry.set(name, 46);
    local.push(header, data); central.push(entry); offset += header.length + data.length; centralSize += entry.length;
  }
  const end = new Uint8Array(22), ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true); ev.setUint16(8, central.length, true); ev.setUint16(10, central.length, true);
  ev.setUint32(12, centralSize, true); ev.setUint32(16, offset, true);
  const result = new Uint8Array(offset + centralSize + 22);
  let cursor = 0;
  for (const chunk of [...local, ...central, end]) { result.set(chunk, cursor); cursor += chunk.length; }
  return result;
}

export function modelExportFiles(config: ZbotConfiguration, format: 'mjcf' | 'urdf', meshes: { ma: string; mb: string }, options: XmlGeneratorOptions = {}): Record<string, string> {
  const files: Record<string, string> = {
    [format === 'mjcf' ? 'model.xml' : 'model.urdf']: format === 'mjcf' ? generateMujocoXML(config, options) : generateUrdf(config),
    'configuration.json': JSON.stringify(config, null, 2),
    'joint-positions.json': JSON.stringify(Object.fromEntries(config.modules.map((m, i) => [`joint_${i}`, (m.initialAngle ?? config.defaultGait.manualAngles[`joint_${i}`] ?? 0) * Math.PI / 180])), null, 2),
    'README.txt': format === 'mjcf'
      ? '解压后加载 model.xml。初始关节角保存在 initial keyframe，MuJoCo 中请重置到该 keyframe。网格应与模型放在同一目录。\n'
      : '解压后加载 model.urdf。网格应与模型放在同一目录。joint-positions.json 保存初始关节位置（弧度），需由使用工具设置。此 URDF 用于运动学与可视化，未提供实测惯量；effort=20 N*m、velocity=10 rad/s 为标称占位值，动力学仿真前请替换并补充惯量。自由基座以 floating joint 表示，部分 ROS 工具不支持此类型。\n',
  };
  if (config.rootConnector) files['README.txt'] += `根连接件为${connectorName(config.rootConnector.type)}，棱长 ${(config.rootConnector.size * 1000).toFixed(2)} mm${config.rootConnector.type === 'tetrahedron' ? '，三角面内切圆直径约 103.92 mm' : ''}。MJCF 使用标称质量 1 kg（非实测）；URDF 仅提供几何与连接拓扑。\n`;
  if (format === 'urdf' && config.rootConnector?.type === 'tetrahedron') files['tetrahedron.obj'] = tetrahedronObj();
  if ((!config.geometryMode || config.geometryMode === 'cad') && config.modules.length) { files['ma.obj'] = meshes.ma; files['mb.obj'] = meshes.mb; }
  return files;
}
