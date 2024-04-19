import meshio
from ..geometry import targets as TG
import numpy as np

def are_meshes_equal(mesh0, mesh1, rtol=1e-5):
    # Check vertices
    ## loop through each cell checking points have correect valsue
    cells0 = mesh0.cells[0].data
    cells1 = mesh1.cells[0].data
    a = np.zeros((cells0.shape[0], 2))
    for i in range(cells0.shape[0]):
        for j in range((cells1).shape[1]):
            a[cells0[i][j]][0] += 1
            a[cells1[i][j]][1] += 1
    print(a)

    b = 0
    for i in range(cells0.shape[0]):
        for j in range((cells1).shape[0]):
            for k in range((mesh1.cells[0].data).shape[1]):
                if np.allclose(mesh0.points[cells0[i][k]],mesh1.points[cells1[j][k]], rtol=rtol):
                    b +=1
                    ## check point data
    assert b > cells0.shape[0]*3
    assert cells0.shape[0] == cells1.shape[0]
    assert 0
                    



def test_rect_reflector():
    reference = meshio.read('data/rect_reflectorref.ply')
    result = TG.rectReflector(0.3, 0.3, 0.006)

    are_meshes_equal(result, reference)


