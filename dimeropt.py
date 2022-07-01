#!/usr/bin/env python
# -*- coding: utf-8 -*-
# dimeropt.py

# Copyright (c) 2006-2022, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Build hollow nanotubes out of DimeroPt.

Dimeropt.py is a Python library to build hollow nanotubes from molecular
coordinates of Pt-Diethynylbiphenyl (DimeroPt).

Refer to reference [1] for details.

:Author: `Christoph Gohlke <http://www.cgohlke.com>`_
:Version: 2022.7.1

Requirements
------------

This release has been tested with the following requirements and dependencies
(other versions may work):

- `Python 2.7 <https://www.python.org>`_
- `Numpy 1.7 <https://pypi.org/project/numpy/>`_
- `PyCifRW 3.5 <https://pypi.org/project/PyCifRW/>`_
- `Matplotlib 1.2 <https://pypi.org/project/matplotlib/>`_
  (optional for plotting)
- `CCDC <https://www.ccdc.cam.ac.uk/structures/>`_
  deposit number 292024 crystallographic information file.
  The filename must be "dimeropt.cif".

References
----------

1. Self-assembly of nanostructured polymetallaynes polymer.
   I Fratoddi, C Gohlke, C Cametti, M Diociaiuti, M V Russo.
   Polymer 49(15), 3211-16, 2008. doi: 10.1016/j.polymer.2008.05.022
2. Platinum (II) dialkynyl bridged binuclear complex and related multinuclear
   oligomer: Comparison of EXAFS and X-ray crystal structure studies.
   C Battocchio, F D'Acapito, I Fratoddi, A La Groia, G Polzonetti,
   G Roviello, M V Russo. Chemical Physics 328(1-3), 269-274, 2006.

"""

from __future__ import division, print_function

__version__ = "2022.7.1"

import math

import numpy
import CifFile


class Molecule(object):
    """Molecule coordinates."""

    def __init__(self, path, fmt=None):
        self.path = path
        self.name = path
        # homogeneous coordinates
        self.atom_pos = numpy.zeros((4, 0), dtype=numpy.float64)
        self.atom_names = []
        self.atom_symbols = []
        self.atom_dict = {}  # atom_dict[name] = [indices]
        self.cell = Cell()

        if not fmt:
            return

        if fmt == "cif":
            cif = CifFile.ReadCif(path)
            cif_data = cif[cif.keys()[0]]
            self.cell = Cell(
                cifstr2float(cif_data["_cell_length_a"]),
                cifstr2float(cif_data["_cell_length_b"]),
                cifstr2float(cif_data["_cell_length_c"]),
                cifstr2float(cif_data["_cell_angle_alpha"]),
                cifstr2float(cif_data["_cell_angle_beta"]),
                cifstr2float(cif_data["_cell_angle_gamma"]),
            )
            self.atom_names = cif_data["_atom_site_label"]
            self.atom_symbols = cif_data["_atom_site_type_symbol"]
            atom_x = cif_data["_atom_site_fract_x"]
            atom_y = cif_data["_atom_site_fract_y"]
            atom_z = cif_data["_atom_site_fract_z"]
            self.atom_pos = numpy.array(
                (
                    cifstr2float(atom_x),
                    cifstr2float(atom_y),
                    cifstr2float(atom_z),
                    [1.0] * len(self),
                ),
                dtype=numpy.float64,
            )
            self.atom_dict = self._atom_dict()
        else:
            raise NotImplementedError("Only CIF files are supported.")

        # Orthogonal coordinates
        self.atom_pos = numpy.dot(self.cell.matrix, self.atom_pos)

    def __len__(self):
        """Return number of atoms in molecule."""
        return len(self.atom_names)

    def __str__(self):
        """Return string containing information about molecule."""
        s = "%s\n%.4f %.4f %.4f %.4f %.4f %.4f" % (
            self.name,
            self.cell.a,
            self.cell.b,
            self.cell.c,
            self.cell.alpha,
            self.cell.beta,
            self.cell.gamma,
        )
        for i in range(len(self)):
            s += "\n%-4i %-5s %-2s %9.6f %9.6f %9.6f" % (
                i,
                self.atom_names[i],
                self.atom_symbols[i],
                self.atom_pos[0][i],
                self.atom_pos[1][i],
                self.atom_pos[2][i],
            )
        return s

    def _atom_dict(self):
        """Return dictionary, mapping atom names to indices."""
        adict = {}
        for i, name in enumerate(self.atom_names):
            adict.setdefault(name, []).append(i)
        return adict

    def save(self, path=None, fmt="xyz"):
        """Save atom coordinates to file."""
        if fmt == "xyz":
            try:
                f = open(path or (self.path + ".xyz"), "w", newline="\n")
            except TypeError:
                # Python 2
                f = open(path or (self.path + ".xyz"), "wb")
            f.write("%i\n%s  0.000000\n" % (len(self), self.name))
            for i in range(len(self)):
                pos = self.atom_pos[:, i]
                f.write(
                    "%-2s %9.6f %9.6f %9.6f\n"
                    % (self.atom_symbols[i], pos[0], pos[1], pos[2])
                )
            f.close()
        else:
            raise NotImplementedError("File format not supported: %s" % fmt)

    def molecular_formula(self):
        """Return molecular formula."""
        elements = {}
        for symbol in self.atom_symbols:
            elements[symbol] = elements.get(symbol, 0) + 1
        formula = []
        for symbol, count in elements.items():
            formula.append(symbol)
            if count > 1:
                formula.append(str(count))
        return "".join(formula)

    def remove_atoms(self, atom_list):
        """Remove multiple atoms from molecule."""
        try:
            atoms = sum([self.atom_dict[atom] for atom in atom_list], [])
        except Exception:
            atoms = atom_list
        try:
            atoms.sort(reverse=True)
        except Exception:
            atoms = (atoms,)
        take = list(range(len(self)))
        for i in atoms:
            del take[i]
            del self.atom_names[i]
            del self.atom_symbols[i]
        self.atom_pos = self.atom_pos.take(take, axis=1)
        self.atom_dict = self._atom_dict()

    def add_atom(self, name="C000", symbol="C", pos=None):
        """Add atom to the molecule."""
        if pos is None:
            pos = [0.0, 0.0, 0.0, 1.0]
        self.atom_names.append(name)
        self.atom_symbols.append(symbol)
        pos.append(1.0)
        self.atom_pos = numpy.concatenate(
            (self.atom_pos, numpy.array(pos).reshape(4, 1)), axis=1
        )
        self.atom_dict.setdefault(name, []).append(len(self) - 1)

    def add_methylene_hydrogens(self, c0, c1, c2):
        """Add two Hydrogen atoms to Carbon C0 of Carbon chain C1-C0-C2."""
        c = self.atom_pos(c0)
        v1 = norm(c - self.atom_pos(c1))
        v2 = norm(c - self.atom_pos(c2))
        s = math.sqrt(1.0 / 3.0) * norm(v1 + v2)
        n = math.sqrt(2.0 / 3.0) * norm(numpy.cross(v1, v2))
        self.add_atom("H0add", "H", c + s + n)
        self.add_atom("H1add", "H", c + s - n)

    def add_methyl_hydrogens(self, c0, c1, c2):
        """Add three Hydrogen atoms to Carbon C0 of Carbon chain C0-C1-C2."""
        c = self.atom_pos(c0)
        v1 = norm(self.atom_pos(c2) - self.atom_pos(c1))
        v2 = norm(c - self.atom_pos(c1))
        s = math.sqrt(1.0 / 3.0) * norm(v1 + v2)
        n = math.sqrt(2.0 / 3.0) * norm(numpy.cross(v1, v2))
        self.add_atom("H0add", "H", c - v1)
        self.add_atom("H1add", "H", c + s + n)
        self.add_atom("H2add", "H", c + s - n)

    def crystalize(self, axes=(0, 1, 2)):
        """Repeat unit cell in all directions."""

        def repeat(self, s):
            T = translation_matrix(self.cell.matrix[:-1, s])
            self.atom_names += self.atom_names
            self.atom_symbols += self.atom_symbols
            new_coords = numpy.dot(T, self.atom_pos)
            self.atom_pos = numpy.concatenate(
                (self.atom_pos, new_coords), axis=1
            )

        for ax in axes:
            repeat(self, ax)
        self.atom_dict = self._atom_dict()

    def transform(self, matrix):
        """Apply transformation matrix to all atoms in place."""
        self.atom_pos = numpy.dot(matrix, self.atom_pos)
        self.cell.transform(matrix)

    def transform_copy(self, matrix, atom_list=None):
        """Apply transformation matrix to copy of specified atoms and
        append them to molecule.

        """
        if atom_list is None:
            # concatenate all atoms
            self.atom_names += self.atom_names
            self.atom_symbols += self.atom_symbols
            new_coords = numpy.dot(matrix, self.atom_pos)
        else:
            # only concatenate specified atoms
            new_coords = numpy.zeros((4, len(atom_list)), dtype=numpy.float64)
            for i, a in enumerate(atom_list):
                self.atom_names.append(self.atom_names[a])
                self.atom_symbols.append(self.atom_symbols[a])
                new_coords[:, i] = self.atom_pos[:, a]
            new_coords = numpy.dot(matrix, new_coords)
        self.atom_pos = numpy.concatenate((self.atom_pos, new_coords), axis=1)
        self.atom_dict = self._atom_dict()

    def sort_atoms_by_distance(self, atom, atom_list):
        """Return atom_list sorted by distance from atom."""
        atom = self.atom_pos[:, atom]
        cmpkey = lambda a: numpy.linalg.norm(atom - self.atom_pos[:, a])
        return sorted(atom_list, key=cmpkey)


class DimeroPt(Molecule):
    """Pt-Diethynylbiphenyl molecule."""

    width = 5.1526  # separation of adjacent molecules in crystal layer
    height = 10.864  # estimated "thickness" of one crystal layer

    def __init__(self):
        """Read dimeropt.cif file and reconstruct monomeric Pt-DEBP molecule.

        The molecule is oriented such that the first Pt atom is positioned
        at the origin, the Pt-Pt axis aligns with the z-axis, and the diphenyl
        mean plane aligns approximately with the yz-plane.

        """
        Molecule.__init__(self, "dimeropt.cif", "cif")

        # remove duplicate atoms
        self.remove_atoms(
            (
                "C2B'",
                "H2B3",
                "H2B4",
                "C3B'",
                "H3B3",
                "H3B4",
                "C4B'",
                "H4B4",
                "H4B5",
                "H4B6",
            )
        )

        # rebuild complete molecule from unit cell.
        # apply -x-y-z symmetry and translate to connect units.
        self.transform_copy(
            numpy.dot(
                translation_matrix(self.cell.matrix[:, 2]),
                scaling_matrix(-1.0, [0, 0, 0]),
            )
        )

        # reposition and reorient molecule
        nv = self._normal()[0:3]
        pt, ax = self._axis()
        pt = pt[0:3]
        ax = unit_vector(ax[0:3])
        self.transform(
            superimpose_matrix(
                [pt, pt + ax, pt + nv], [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
            )
        )

    def _axis(self):
        """Return axis pointing from first to last Pt atom."""
        pts = self.atom_dict['Pt1']
        pt0 = self.atom_pos[:, pts[0]]
        pts = self.sort_atoms_by_distance(pts[0], pts)
        pt1 = self.atom_pos[:, pts[-1]]
        return pt0.copy(), pt1 - pt0

    def _normal(self):
        """Return vector approximately normal to the two aromatic rings and
        perpendicular to Pt-Pt axis.

        """
        # normal vector to aromatic rings
        d = self.atom_dict
        p = self.atom_pos
        a = p[:, d['H12'][0]] - p[:, d['H12'][1]]
        b = p[:, d['H9'][0]] - p[:, d['H9'][1]]
        x = a[1] * b[2] - a[2] * b[1]
        y = a[2] * b[0] - a[0] * b[2]
        z = a[0] * b[1] - a[1] * b[0]
        # make normal vector perpendicular to Pt-Pt axis
        _, ax = self._axis()
        ax = unit_vector(ax[0:3])
        nv = numpy.cross(ax, numpy.cross([x, y, z], ax))
        return unit_vector((nv[0], nv[1], nv[2], 0.0))

    def duplicate(self):
        """Duplicate length of molecule using point symmetry at Pt."""
        # find Pt atom most distant from first Pt atom
        pts = self.atom_dict["Pt1"]
        pts = self.sort_atoms_by_distance(pts[0], pts)
        pt = pts[-1]
        pt_pos = self.atom_pos[:, pt]
        # list of atoms to duplicate
        exclude = list(range(pt, pt + 4)) + list(range(pt + 16, pt + 94))
        include = [x for x in range(len(self)) if x not in exclude]
        # point symmetry at last Pt
        self.transform_copy(scaling_matrix(-1.0, pt_pos), include)
        # remove terminal Cl
        self.remove_atoms(pt + 2)

    def add_adjacent_molecule(self):
        """Add adjacent molecule."""
        self.transform_copy(translation_matrix(self.cell.matrix[:-1, 0]))

    def tube_diameter(self, n):
        """Return radius in A of nanotube of n polymer molecules."""
        r = self.width / 2.0 / math.tan(math.pi / float(n))
        r += self.height  # extend inner radius by the thickness of a molecule
        return r * 2.0

    def tube_molecules(self, d):
        """Return number of polymer molecules in nanotube of diameter in A."""
        r = d / 2.0 - self.height
        return math.pi / math.atan(self.width / 2.0 / r)

    def tubify(self, duplicate=1, m=28):
        """Construct a hollow nanotube.

        Generate hollow nanotube consisting of an even number m of oligomers.
        The oligomer molecule is translated by w/2+v/(2 tan(p/m)) along the
        x-axis and copied m times. The ith copy is rotated by 2pi/m around
        the z axis and every second molecule is translated by u/2 along the
        zaxis. The dimensions u, v, and w are derived from the crystal
        structure of the binuclear Pt dialkynyl bridged complex:

        - u=16.343 A: the distance between the two Pt atoms of a molecule.
        - v=5.153 A: the distance between the Pt-Pt axes of two parallel
          molecules in a layer of molecules.
        - w=10.864 A: the shortest distance between the diphenyl planes of
          two molecules in separate layers of molecules.

        """
        if (m < 4) or (divmod(m, 2)[1] != 0):
            raise ValueError(
                "Nanotubes can only be contructed from four "
                "or more equal number of polymer molecules."
            )
        radius = self.height / 2.0 + self.width / (2.0 * math.tan(math.pi / m))
        # rotation axis
        point, direction = self._axis()
        assert not numpy.allclose(direction, 0.0)
        point += self._normal() * radius
        # extend molecule to length
        for i in range(duplicate):
            self.duplicate()
        # adjacent molecules are offset half a molecule size along axis
        t = translation_matrix(direction / 2.0)
        # rotation around axis
        atomlist = list(range(len(self)))
        for i in range(1, m):
            angle = math.degrees(2.0 * math.pi * i / m)
            R = rotation_matrix(angle, direction, point)
            if divmod(i, 2)[1]:
                # translate every second molecule
                R = numpy.dot(R, t)
            self.transform_copy(R, atomlist)
        return radius


class Cell(object):
    """Store lengths and angles of a crystallographic unit cell."""

    def __init__(
        self, a=10.0, b=10.0, c=10.0, alpha=90.0, beta=90.0, gamma=90.0
    ):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.matrix = self._orthogonalization_matrix()

    def __str__(self):
        f = "a: %.4f\nb: %.4f\nc: %.4f\nalpha: %.4f\nbeta:  %.4f\ngamma: %.4f"
        return f % (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    def _orthogonalization_matrix(self):
        """Return orthogonalization matrix."""
        al = math.radians(self.alpha)
        be = math.radians(self.beta)
        ga = math.radians(self.gamma)
        sia = math.sin(al)
        sib = math.sin(be)
        coa = math.cos(al)
        cob = math.cos(be)
        cog = math.cos(ga)
        co = (coa * cob - cog) / (sia * sib)
        return numpy.array(
            (
                (self.a * sib * math.sqrt(1.0 - co * co), 0.0, 0.0, 0.0),
                (-self.a * sib * co, self.b * sia, 0.0, 0.0),
                (self.a * cob, self.b * coa, self.c, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=numpy.float64,
        )

    def transform(self, matrix):
        """Transform unit cell using homogeneous transformation matrix."""
        self.matrix = numpy.dot(matrix, self.matrix)


def translation_matrix(direction):
    """Return matrix to translate by direction vector."""
    M = numpy.identity(4, dtype=numpy.float64)
    M[0:3, 3] = direction[0:3]
    return M


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction."""
    M = numpy.identity(4, dtype=numpy.float64)
    a = math.radians(angle)
    u = numpy.array(direction[0:3], dtype=numpy.float64, copy=True)
    u /= math.sqrt(numpy.dot(u, u))  # unit vector of direction
    # rotation matrix around unit vector
    R = (
        numpy.identity(3, dtype=numpy.float64) * math.cos(a)
        + numpy.outer(u, u) * (1.0 - math.cos(a))
        + math.sin(a)
        * numpy.array(
            [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]],
            dtype=numpy.float64,
        )
    )
    M[0:3, 0:3] = R
    if point is not None:
        # rotation not around origin
        M[0:3, 3] = point[0:3] - numpy.dot(R, point[0:3])
    return M


def scaling_matrix(factor, origin=None, direction=None):
    """Return matrix to scale by factor around origin in direction."""
    if origin is None:
        origin = numpy.zeros((3,), dtype=numpy.float64)
    else:
        origin = numpy.array(origin[0:3], dtype=numpy.float64, copy=False)
    if direction is None:
        # uniform scaling
        M = numpy.identity(4, dtype=numpy.float64)
        M *= factor
        M[0:3, 3] = (1.0 - factor) * origin
        M[3, 3] = 1.0
    else:
        # nonuniform scaling
        M = numpy.identity(4, dtype=numpy.float64)
        direction = numpy.array(direction[0:3], dtype=numpy.float64, copy=True)
        direction /= math.sqrt(numpy.dot(direction, direction))
        M[0:3, 0:3] -= (1.0 - factor) * numpy.outer(direction, direction)
        M[0:3, 3] = ((1.0 - factor) * numpy.dot(origin, direction)) * direction
    return M


def unit_vector(vector, out=None):
    """Return vector normalized by its length."""
    if out is None:
        out = numpy.array(vector, dtype=numpy.float64, copy=True)
        out /= math.sqrt(numpy.dot(out, out))
        return out
    else:
        out[:] = numpy.array(vector, dtype=numpy.float64, copy=False)
        out /= math.sqrt(numpy.dot(out, out))


def superimpose_matrix(v0, v1):
    """Return matrix to transform given vector set to second vector set."""
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)[:, :3]
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)[:, :3]
    if v0.shape != v1.shape or v0.shape[0] < 3:
        raise ValueError("Vector sets are of wrong shape or type.")
    t0 = numpy.mean(v0, axis=0)
    t1 = numpy.mean(v1, axis=0)
    v0 = v0 - t0
    v1 = v1 - t1
    u, _, vh = numpy.linalg.svd(numpy.dot(v1.T, v0))
    R = numpy.dot(u, vh)
    if numpy.linalg.det(R) < 0.0:
        R -= numpy.outer(u[:, 2], vh[2, :] * 2.0)
    M = numpy.identity(4, dtype=numpy.float64)
    T = numpy.identity(4, dtype=numpy.float64)
    M[0:3, 0:3] = R
    T[0:3, 3] = t1
    M = numpy.dot(T, M)
    T[0:3, 3] = -t0
    return numpy.dot(M, T)


def norm(vector):
    """Return length of vector, i.e. its euclidean norm."""
    return numpy.sqrt(numpy.dot(vector, vector))


def cifstr2float(cif):
    """Convert CIF string to float, discarding precisions."""
    try:  # scalar
        return float(cif.split("(", 1)[0])
    except AttributeError:  # list
        return [float(n.split("(", 1)[0]) for n in cif]


def examples():
    """Generate structures from DimeroPt molecule."""
    monomer = DimeroPt()
    monomer.save("dimeropt_monomer.xyz")

    crystal = DimeroPt()
    crystal.crystalize()
    crystal.save("dimeropt_crystal.xyz")

    polymer = DimeroPt()
    polymer.duplicate()
    polymer.save("dimeropt_polymer.xyz")

    # build nanotube of ~6.8 nm diameter shown in figure 5 of reference [1]
    d = 68.0
    m = int(divmod(round(polymer.tube_molecules(d)), 2.0)[0] * 2)
    tube = DimeroPt()
    tube.tubify(duplicate=1, m=m)
    tube.save("dimeropt_tube_of_%i.xyz" % m)
    print("Molecules: %i" % m)
    print("Diameter: %.3f nm" % (tube.tube_diameter(m) * 10))
    print("Formula: %s" % tube.molecular_formula())
    print("Atoms: %i" % len(tube))

    def plot_nanotube_diameters():
        from matplotlib import pyplot

        data = [(n, tube.tube_diameter(n) / 10.0) for n in range(22, 50, 2)]
        data = numpy.array(data)
        pyplot.plot(data[:, 0], data[:, 1], "o-")
        pyplot.title("Nanotube Diameter (nm) vs. Number of Polymers")
        pyplot.show()

    plot_nanotube_diameters()


if __name__ == "__main__":
    examples()
