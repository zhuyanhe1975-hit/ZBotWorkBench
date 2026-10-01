import assert from 'node:assert/strict';
import { before, test } from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import { createConfiguration, createCubeConfiguration, createCubeQuadruped, createTetrahedronConfiguration, parseConfiguration } from '../src/utils/configuration';
import { PRESET_CONFIGURATIONS } from '../src/data/presets';
import { SEVEN_DOF_PRESETS } from '../src/data/sevenDofPresets';
import { generateZbotOBJ } from '../src/utils/zbotMeshGenerator';
import { generateMujocoXML } from '../src/utils/xmlGenerator';
import { generateUrdf } from '../src/utils/urdfGenerator';
import { saveConfigurationToLibrary, readConfigurationLibrary } from '../src/utils/configurationLibrary';

let mj: Awaited<ReturnType<typeof loadMujoco>>;
before(async () => { mj = await loadMujoco(); mj.FS.writeFile('ma.obj', generateZbotOBJ('ma')); mj.FS.writeFile('mb.obj', generateZbotOBJ('mb')); });

for (const create of [createConfiguration, createCubeConfiguration, createTetrahedronConfiguration, createCubeQuadruped, ...PRESET_CONFIGURATIONS.map(config => () => structuredClone(config)), ...SEVEN_DOF_PRESETS.map(config => () => structuredClone(config))]) {
  for (const mode of ['fixed', 'free'] as const) {
    test(`${create().name} root ${mode} preserves mode and obeys root dynamics`, () => {
      const config = create(); config.baseMode = mode; config.rootPos = [0, 0, 1];
      assert.equal(parseConfiguration(JSON.stringify(config)).baseMode, mode);
      const values = new Map<string, string>();
      const storage = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
      saveConfigurationToLibrary(storage, config);
      assert.equal(readConfigurationLibrary(storage)[0].baseMode, mode);
      const xml = generateMujocoXML(config);
      assert.equal(xml.includes('<freejoint/>'), mode === 'free');
      assert.match(generateUrdf(config), new RegExp(`<joint name="base_joint" type="${mode === 'free' ? 'floating' : 'fixed'}">`));
      const model = mj.MjModel.from_xml_string(xml), data = new mj.MjData(model);
      try {
        assert.equal(model.nq, config.modules.length + (mode === 'free' ? 7 : 0));
        assert.equal(model.nv, config.modules.length + (mode === 'free' ? 6 : 0));
        mj.mj_forward(model, data);
        const height = data.xpos[5];
        for (let i = 0; i < 50; i++) mj.mj_step(model, data);
        if (mode === 'free') { assert.ok(data.xpos[5] < height); assert.ok(data.qvel[2] < 0); }
        else assert.equal(data.xpos[5], height);
      } finally { data.delete(); model.delete(); }
    });
  }
}
