"""
Mesh data structures for spatial discretization.

Supports regular 2D/3D grids initially, with interface ready
for unstructured meshes (finite volume, finite element, MOC flat-source regions).
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Cell:
    """A single computational cell / mesh element."""
    id: int
    volume: float
    material_id: str
    centroid: tuple[float, ...]
    state: "CellState | None" = None  # forward ref


@dataclass
class Face:
    """Interface between two cells (or cell and boundary)."""
    id: int
    left_cell: int
    right_cell: int | None   # None = boundary
    area: float
    distance: float           # centroid-to-centroid distance across this face
    boundary_type: str = "vacuum"  # vacuum | reflective | periodic


@dataclass
class Mesh:
    """Abstract mesh: list of cells and faces.

    Subclasses implement specific geometry (regular grid, unstructured, etc.).
    """
    cells: list[Cell]
    faces: list[Face]
    dim: int

    def n_cells(self) -> int:
        return len(self.cells)

    def n_faces(self) -> int:
        return len(self.faces)

    def get_cell(self, cell_id: int) -> Cell:
        return self.cells[cell_id]

    def get_faces_of_cell(self, cell_id: int) -> list[Face]:
        """Return faces where left_cell == cell_id or right_cell == cell_id."""
        result = []
        for f in self.faces:
            if f.left_cell == cell_id or f.right_cell == cell_id:
                result.append(f)
        return result

    def get_neighbor(self, face: Face, cell_id: int) -> int | None:
        """Given a face and one side, return the other side's cell id (or None for boundary)."""
        if face.left_cell == cell_id:
            return face.right_cell
        elif face.right_cell == cell_id:
            return face.left_cell
        return None


class RegularGrid2D(Mesh):
    """2D regular Cartesian grid with uniform spacing.

    Cell numbering: row-major, i = y * nx + x.
    Faces: east-west (x-faces) then north-south (y-faces).
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 1.0,
        dy: float = 1.0,
        material_map: list[list[str]] | None = None,
        boundary_x: str = "vacuum",
        boundary_y: str = "vacuum",
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y

        cells = []
        faces = []
        face_id = 0

        # --- Cells ---
        for j in range(ny):
            for i in range(nx):
                cell_id = j * nx + i
                mat_id = "default"
                if material_map is not None:
                    mat_id = material_map[j][i]
                cells.append(Cell(
                    id=cell_id,
                    volume=dx * dy,
                    material_id=mat_id,
                    centroid=(i * dx + dx / 2, j * dy + dy / 2),
                ))

        # --- X-direction faces (east-west) ---
        # For nx cells in x, there are nx+1 vertical face columns.
        # Face (j, i) separates cell (j,i-1) and cell (j,i) [left/right].
        for j in range(ny):
            for i in range(nx + 1):
                if i == 0:
                    left, right = None, j * nx + i
                    btype = boundary_x
                elif i == nx:
                    left, right = j * nx + (i - 1), None
                    btype = boundary_x
                else:
                    left, right = j * nx + (i - 1), j * nx + i
                    btype = "internal"
                faces.append(Face(
                    id=face_id,
                    left_cell=left if left is not None else -1,
                    right_cell=right,
                    area=dy,
                    distance=dx,
                    boundary_type=btype,
                ))
                face_id += 1

        # Fix left_cell=-1 sentinel back to None
        for f in faces:
            if f.left_cell == -1:
                f.left_cell = None

        # --- Y-direction faces (north-south) ---
        for j in range(ny + 1):
            for i in range(nx):
                if j == 0:
                    left, right = None, j * nx + i
                    btype = boundary_y
                elif j == ny:
                    left, right = (j - 1) * nx + i, None
                    btype = boundary_y
                else:
                    left, right = (j - 1) * nx + i, j * nx + i
                    btype = "internal"
                faces.append(Face(
                    id=face_id,
                    left_cell=left if left is not None else -1,
                    right_cell=right,
                    area=dx,
                    distance=dy,
                    boundary_type=btype,
                ))
                face_id += 1

        for f in faces:
            if f.left_cell == -1:
                f.left_cell = None

        super().__init__(cells=cells, faces=faces, dim=2)

    def get_cell_ij(self, i: int, j: int) -> Cell:
        return self.cells[j * self.nx + i]

    def get_material_map(self) -> np.ndarray:
        """Return (ny, nx) array of material_id strings."""
        mat = np.empty((self.ny, self.nx), dtype=object)
        for j in range(self.ny):
            for i in range(self.nx):
                mat[j, i] = self.cells[j * self.nx + i].material_id
        return mat

    def to_dict(self) -> dict:
        return {
            "type": "regular_2d",
            "nx": self.nx,
            "ny": self.ny,
            "dx": self.dx,
            "dy": self.dy,
            "n_cells": self.n_cells(),
            "n_faces": self.n_faces(),
        }
