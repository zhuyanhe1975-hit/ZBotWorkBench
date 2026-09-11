"""Capture PhysX's cooked convex meshes from an already running Isaac stage.

This helper never starts SimulationApp or a GPU process. Call
``capture_cooked_colliders(env.scene.stage)`` from the reference capture script.
Vertices are expressed in each rigid body's local frame, so exported collision
geometry can be compared directly with the browser's MJCF geometry.
"""
import re
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils


def capture_cooked_colliders(stage, root_path=None):
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult

    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    if not stage_id.IsValid():
        stage_id = cache.Insert(stage)
    cooking = get_physx_cooking_interface()
    records = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if root_path and not (prim_path == root_path or prim_path.startswith(root_path.rstrip('/') + '/')):
            continue
        # Parallel training replicas have identical geometry; capture env_0 only.
        replica = re.search(r'/env_(\d+)(?:/|$)', prim_path)
        if replica and replica.group(1) != '0':
            continue
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim.GetParent()
        while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body or body.IsPseudoRoot():
            continue
        result = {'primPath': prim_path, 'bodyPath': str(body.GetPath()), 'bodyName': body.GetName(),
                  'status': 'pending', 'hulls': []}
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        result['approximation'] = str(approximation)
        if approximation not in ('convexHull', 'convexDecomposition'):
            result.update(status='failed', error=f'Unsupported approximation {approximation}')
            records.append(result)
            continue
        transform = (UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                     * UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse())
        result['meshToBodyTransform'] = [[float(transform[i][j]) for j in range(4)] for i in range(4)]

        def received(status, convexes):
            result['result'] = str(status)
            if status != PhysxCollisionRepresentationResult.RESULT_VALID:
                result.update(status='failed', error=f'PhysX cooking returned {status}')
                return
            try:
                hulls = []
                for convex in convexes:
                    vertices = []
                    for vertex in convex.vertices:
                        point = transform.Transform(Gf.Vec3d(vertex.x, vertex.y, vertex.z))
                        vertices.append([float(point[i]) for i in range(3)])
                    faces = []
                    for polygon in convex.polygons:
                        indices = convex.indices[polygon.index_base:polygon.index_base + polygon.num_vertices]
                        for i in range(1, len(indices) - 1):
                            faces.append([int(indices[0]), int(indices[i]), int(indices[i + 1])])
                    if not vertices or not faces:
                        raise ValueError('Cooked hull has no vertices or triangle faces')
                    if any(i < 0 or i >= len(vertices) for face in faces for i in face):
                        raise ValueError('Cooked hull contains invalid vertex indices')
                    hulls.append({'vertices': vertices, 'faces': faces})
                if not hulls:
                    raise ValueError('PhysX returned no convex hulls')
                result.update(status='completed', hulls=hulls)
            except Exception as error:
                result.update(status='failed', error=str(error))

        try:
            cooking.request_convex_collision_representation(
                stage_id=stage_id.ToLongInt(),
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim_path),
                run_asynchronously=False,
                on_result=received,
            )
            if result['status'] == 'pending':
                result.update(status='failed', error='Synchronous cooking did not invoke its callback')
        except Exception as error:
            result.update(status='failed', error=str(error))
        records.append(result)
    errors = [f"{record['primPath']}: {record['error']}" for record in records if record['status'] != 'completed']
    if not records:
        errors.append('No rigid-body mesh colliders found in the requested stage subtree')
    return {'version': 1, 'coordinateFrame': 'rigid-body-local', 'colliders': records,
            'available': bool(records) and not errors, 'errors': errors}
