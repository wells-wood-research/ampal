"""Tools for creating a frame dataset.

In this type of dataset, all individual entries are stored separately in a flat
structure.
"""

import pathlib
import typing as t
import warnings
from dataclasses import dataclass

import numpy as np

from .amino_acids import (
    modified_amino_acids,
    polarity_Zimmerman,
    residue_charge,
    standard_amino_acids,
)
from .assembly import Assembly
from .base_ampal import Atom
from .geometry import angle_between_vectors, dihedral
from .protein import Polypeptide, Residue

ATOM_VANDERWAAL_RADII = {
    # Atomic number : Radius
    0: 0.7,  # Carbon
    1: 0.65,  # Nitrogen
    2: 0.6,  # Oxygen
}
PDB_REQUEST_URL = "https://files.rcsb.org/download/"


# {{{ Types
@dataclass
class ResidueResult:
    residue_id: str
    label: str
    encoded_residue: np.ndarray
    data: np.ndarray
    voxels_as_gaussian: bool
    rotamers: str


@dataclass
class DatasetMetadata:
    make_frame_dataset_ver: str
    frame_dims: t.Tuple[int, int, int, int]
    atom_encoder: t.List[str]
    encode_cb: bool
    atom_filter_fn: str
    residue_encoder: t.List[str]
    frame_edge_length: float
    voxels_as_gaussian: bool

    @classmethod
    def import_metadata_dict(cls, meta_dict: t.Dict[str, t.Any]):
        """
        Imports metadata of a dataset from a dictionary object to the DatasetMetadata class.

        Parameters
        ----------
        meta_dict: t.Dict[str, t.Any]
            Dictionary of metadata parameters for the dataset.

        Returns
        -------
        DatasetMetadata dataclass with filled metadata.

        """
        return cls(**meta_dict)


StrOrPath = t.Union[str, pathlib.Path]
ChainDict = t.Dict[str, t.List[ResidueResult]]
# }}}


# {{{ Residue Frame Creation
class Codec:
    def __init__(self, atomic_labels: t.List[str]):
        # Set attributes:
        self.atomic_labels = atomic_labels
        self.encoder_length = len(self.atomic_labels)
        self.label_to_encoding = dict(
            zip(self.atomic_labels, range(self.encoder_length))
        )
        self.encoding_to_label = dict(
            zip(range(self.encoder_length), self.atomic_labels)
        )
        return

    # Labels Class methods:
    @classmethod
    def CNO(cls):
        return cls(["C", "N", "O"])

    @classmethod
    def CNOCB(cls):
        return cls(["C", "N", "O", "CB"])

    @classmethod
    def CNOCACB(cls):
        return cls(["C", "N", "O", "CA", "CB"])

    @classmethod
    def CNOCACBQ(cls):
        return cls(["C", "N", "O", "CA", "CB", "Q"])

    @classmethod
    def CNOCACBP(cls):
        return cls(["C", "N", "O", "CA", "CB", "P"])

    def encode_atom(self, atom_label: str) -> np.ndarray:
        """
        Encodes atoms in a boolean array depending on the type of encoding chosen.

        Parameters
        ----------
        atom_label: str
            Label of the atom to be encoded.

        Returns
        -------
        atom_encoding: np.ndarray
            Boolean array with atom encoding of shape (encoder_length,)

        """
        # Creating empty atom encoding:
        encoded_atom = np.zeros(self.encoder_length, dtype=bool)
        # Attempt encoding:
        if atom_label in self.label_to_encoding.keys():
            atom_idx = self.label_to_encoding[atom_label]
            encoded_atom[atom_idx] = True
        # Encode CA as C in case it is not in the labels:
        elif atom_label == "CA":
            atom_idx = self.label_to_encoding["C"]
            encoded_atom[atom_idx] = True
        else:
            warnings.warn(
                f"{atom_label} not found in {self.atomic_labels} encoding. Returning None."
            )

        return encoded_atom

    def encode_gaussian_atom(
        self, atom_label: str, modifiers_triple: t.Tuple[float, float, float]
    ) -> t.Tuple[np.ndarray, int]:
        """
        Encodes atom as a 3x3 gaussian with length of encoder length. Only the
        C, N and O atoms are represented in gaussian form. If Ca and Cb are
        encoded separately, they will be represented as a gaussian in their
        separate channel.

        Parameters
        ----------
        atom_label: str
            Label of the atom to be encoded.
        modifiers_triple: t.Tuple[float, float, float]
            Triple of the difference between the discretized coordinate and the
            non-discretized coordinate.

        Returns
        -------
        atom_encoding: np.ndarray
            Boolean array with atom encoding of shape (3, 3, 3, encoder_length,)

        """
        atom_idx = -1000
        # Creating empty atom encoding:
        encoded_atom = np.zeros((3, 3, 3, self.encoder_length))
        # Attempt encoding:
        if atom_label.upper() in self.label_to_encoding.keys():
            atom_idx = self.label_to_encoding[atom_label.upper()]
        # Fix labels:
        elif self.encoder_length == 3:
            # In this scenario, the encoder is C,N,O so Cb and Ca are just C atoms
            if (atom_label.upper() == "CA") or (atom_label.upper() == "CB"):
                atom_label = "C"
                atom_idx = self.label_to_encoding[atom_label]
        elif self.encoder_length == 4:
            # In this scenario, Ca is a carbon atom but Cb has a separate channel
            if atom_label.upper() == "CA":
                atom_label = "C"
                atom_idx = self.label_to_encoding[atom_label]
        else:
            raise ValueError(
                f"{atom_label} not found in {self.atomic_labels} encoding."
            )
        # If label to encode is C, N, O:
        if atom_idx in ATOM_VANDERWAAL_RADII.keys():
            # Get encoding:
            atomic_radius = ATOM_VANDERWAAL_RADII[atom_idx]
            atom_to_encode = convert_atom_to_gaussian_density(
                modifiers_triple, atomic_radius
            )
            # Add to original atom:
            encoded_atom[:, :, :, atom_idx] += atom_to_encode
        # If label encodes Cb and Ca as separate channels (ie, not CNO):
        elif atom_label.upper() in self.label_to_encoding.keys():
            # Get encoding:
            atomic_radius = ATOM_VANDERWAAL_RADII[0]
            atom_to_encode = convert_atom_to_gaussian_density(
                modifiers_triple, atomic_radius
            )
            encoded_atom[:, :, :, atom_idx] += atom_to_encode
        else:
            raise ValueError(
                f"{atom_label} not found in {self.atomic_labels} encoding. Returning empty array."
            )

        return encoded_atom, atom_idx

    def decode_atom(self, encoded_atom: np.ndarray) -> t.Optional[str]:
        """
        Decodes atoms into string depending on the type of encoding chosen.

        Parameters
        ----------
        encoded_atom: np.ndarray
            Boolean array with atom encoding of shape (encoder_length,)

        Returns
        -------
        decoded_atom: t.Optional[str]
            Label of the decoded atom.
        """
        # Get True index of one-hot encoding
        atom_encoding = np.nonzero(encoded_atom)[0]

        if atom_encoding.size == 0:
            warnings.warn("Encoded atom was 0.")
        # If not Empty space:
        else:
            # Decode Atom:
            decoded_atom = self.encoding_to_label[atom_encoding[0]]
            return decoded_atom


def align_to_residue_plane(residue: Residue):
    """Reorients the parent ampal.Assembly that the peptide plane lies on xy.

    Notes
    -----
    This changes the assembly **in place**.

    Parameters
    ----------
    residue: ampal.Residue
        Residue that will be used as a reference to reorient the assemble.
        The assembly will be reoriented so that the residue['CA'] lies on the origin,
        residue['N'] lies on +Y and residue['C'] lies on +X-Y, assuming correct
        geometry.

    Raises
    ------
    ZeroDivisionError:
        If
    """
    # unit vectors used for alignment
    origin = (0, 0, 0)
    unit_y = (0, 1, 0)
    unit_x = (1, 0, 0)

    # translate the whole parent assembly so that residue['CA'] lies on the origin
    translation_vector = residue["CA"].array
    if residue.parent:
        if residue.parent.parent:
            assembly = residue.parent.parent
        else:
            raise ValueError(
                "Residue does not have a parent assembly that will be aligned"
            )
    else:
        raise ValueError("Residue does not have a parent Polymer.")
    assembly.translate(-translation_vector)

    # rotate whole assembly so that N-CA lies on Y
    n_vector = residue["N"].array
    rotation_axis = np.cross(n_vector, unit_y)
    try:
        assembly.rotate(angle_between_vectors(n_vector, unit_y), rotation_axis)
    except ZeroDivisionError:
        pass

    # align C with xy plane
    rotation_angle = dihedral(unit_x, origin, unit_y, residue["C"])
    assembly.rotate(-rotation_angle, unit_y)

    return


def encode_cb_to_ampal_residue(residue: Residue):
    """
    Encodes a Cb atom to an AMPAL residue. The Cb is added to an average position
    calculated by averaging the Cb coordinates of the aligned frames for the 1QYS protein.

    Parameters
    ----------
    residue: ampal.Residue
        Focus residues that requires the Cb atom.

    """
    avg_cb_position = (-0.741287356, -0.53937931, -1.224287356)
    cb_atom = Atom(avg_cb_position, element="C", res_label="CB", parent=residue)
    residue["CB"] = cb_atom
    return


def encode_cb_before_voxelization(residue: Residue):
    """
    Encodes a Cb atom to all of the AMPAL residues before the voxelization begins. The Cb is added to an average position
    calculated by averaging the Cb coordinates of the aligned frames for the 1QYS protein.

    Parameters
    ----------
    residue: ampal.Residue
        Focus residues that requires the Cb atom.

    """
    align_to_residue_plane(residue)
    encode_cb_to_ampal_residue(residue)
    return


def within_frame(frame_edge_length: float, atom: Atom) -> bool:
    """Tests if an atom is within the `frame_edge_length` of the origin."""
    half_frame_edge_length = frame_edge_length / 2
    return all([0 <= abs(v) <= half_frame_edge_length for v in atom.array])


def discretize(
    atom: Atom, voxel_edge_length: float, adjust_by: int = 0
) -> t.Tuple[int, int, int]:
    """Rounds and then converts to an integer.

    Parameters
    ----------
    atom: ampal.Atom
        Atom x, y, z coordinates will be discretized based on `voxel_edge_length`.
    voxel_edge_length: float
        Edge length of the voxels that are mapped onto cartesian space.
    adjust_by: int
    """

    # I'm explicitly repeating this operation to make it explicit to the type checker
    # that a triple is returned.
    return (
        int(np.round(atom.x / voxel_edge_length)) + adjust_by,
        int(np.round(atom.y / voxel_edge_length)) + adjust_by,
        int(np.round(atom.z / voxel_edge_length)) + adjust_by,
    )


def encode_residue(residue: str) -> np.ndarray:
    """
    One-Hot Encodes a residue string to a numpy array. Attempts to convert non-standard
    residues using AMPAL's UNCOMMON_RESIDUE_DICT.

    Parameters
    ----------
    residue: str
        Residue label of the frame.

    Returns
    -------
    residue_encoding: np.ndarray
        One-Hot encoding of the residue with shape (20,)
    """
    std_residues = list(standard_amino_acids.values())
    residue_encoding = np.zeros(len(std_residues), dtype=bool)

    # Deal with non-standard residues:
    if residue not in std_residues:
        if residue in modified_amino_acids.keys():
            warnings.warn(f"{residue} is not a standard residue.")
            residue_label = modified_amino_acids[residue]
            warnings.warn(f"Residue converted to {residue_label}.")
        else:
            raise ValueError(
                f"Expected natural amino acid, attempted conversion from uncommon residues, but got {residue}."
            )
    else:
        residue_label = residue

    # Add True at the correct residue index:
    res_idx = std_residues.index(residue_label)
    residue_encoding[res_idx] = 1

    return residue_encoding


def convert_atom_to_gaussian_density(
    modifiers_triple: t.Tuple[float, float, float],
    vanderwaal_radius: float,
    range_val: int = 2,
    resolution: int = 999,
    optimized: bool = True,
):
    """
    Converts an atom at a coordinate, with specific modifiers due to a discretization,
    into a 3x3x3 gaussian density using the formula indicated  by Zhang et al., (2019) ProdCoNN.

    https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fprot.25868&file=prot25868-sup-0001-AppendixS1.pdf

    Parameters
    ----------
    modifiers_triple: t.Tuple[float, float, float]
        Triple of the difference between the discretized coordinate and the non-discretized coordinate.
    vanderwaal_radius: float
        Van der Waal radius of the current atom being discretized.
    range_val: int
        Range values for the gaussian. Default = 2
    resolution: int
        Number of points selected between the `range_val` to calculate the density. Default = 999
    optimized: bool
        Whether to use an optimized algorithm to produce the gaussian. Default = True
    Returns
    -------
    norm_gaussian_frame: np.ndarray
        3x3x3 Frame encoding for a gaussian atom
    """
    if optimized:
        # Unpack x, y, z:
        x, y, z = modifiers_triple
        # Obtain x, y, z ranges for the gaussian
        x_range = np.linspace(-range_val + x, range_val - x, resolution)
        y_range = np.linspace(-range_val + y, range_val - y, resolution)
        z_range = np.linspace(-range_val + z, range_val - z, resolution)
        # Calculate density for each axis at each point
        x_vals, y_vals, z_vals = (
            np.exp(-1 * ((x_range - x) / vanderwaal_radius) ** 2, dtype=np.float16),
            np.exp(-1 * ((y_range - y) / vanderwaal_radius) ** 2, dtype=np.float16),
            np.exp(-1 * ((z_range - z) / vanderwaal_radius) ** 2, dtype=np.float16),
        )

        x_densities = []
        y_densities = []
        z_densities = []
        r = resolution // 3
        # Integrate to get area under the gaussian curve:
        for i in range(0, resolution, r):
            x_densities.append(np.trapz(x_vals[i : i + r], x_range[i : i + r]))
            y_densities.append(np.trapz(y_vals[i : i + r], y_range[i : i + r]))
            z_densities.append(np.trapz(z_vals[i : i + r], z_range[i : i + r]))
        # # Create grids for x, y and z :
        xyz_grids = np.meshgrid(x_densities, y_densities, z_densities)
        # The multiplication here is necessary so that e**x * e**y * e**z are equivalent to
        # e**(x + y + z)
        gaussian_frame = xyz_grids[0] * xyz_grids[1] * xyz_grids[2]

    else:
        gaussian_frame = np.zeros((3, 3, 3), dtype=float)
        xyz_coordinates = np.where(gaussian_frame == 0)
        # Identify the real (non-discretized) coordinates of the atom in the 3x3x3 matrix
        x, y, z = modifiers_triple
        x, y, z = x + 1, y + 1, z + 1
        # The transpose changes arrays of [y], [x], [z] into [y, x, z]
        for voxel_coord in np.array(xyz_coordinates).T:
            # Extract voxel coords:
            vy, vx, vz = voxel_coord
            # Calculate Density:
            voxel_density = np.exp(
                -((vx - x) ** 2 + (vy - y) ** 2 + (vz - z) ** 2) / vanderwaal_radius**2
            )
            # Add density to frame:
            gaussian_frame[vy, vx, vz] = voxel_density

    # Normalize so that values add up to 1:
    norm_gaussian_frame = gaussian_frame / np.sum(gaussian_frame)

    return norm_gaussian_frame


def calculate_atom_coord_modifier_within_voxel(
    atom: Atom,
    voxel_edge_length: float,
    indices: t.Tuple[int, int, int],
    adjust_by: int = 0,
) -> t.Tuple[float, float, float]:
    """
    Calculates modifiers lost during the discretization of atoms within the voxel.

    Assuming an atom coordinate was x = 2.6, after the discretization it will occupy
    the voxel center_voxel_x + int(2.6 / 0.57). So assuming center voxel of 10,
    the atom will occupy x = 10 + 5, though (2.6 / 0.57) is about 4.56.

    This means that our discretization is losing about 5 - 4.56 = 0.44 position.
    The gaussian representation aims at accounting for this approximation.

    Parameters
    ----------
    atom: ampal.Atom
        Atom x, y, z coordinates will be discretized based on `voxel_edge_length`.
    voxel_edge_length: float
        Edge length of the voxels that are mapped onto cartesian space.
    indices: t.Tuple[int, int, int],
        Triple containing discretized (x, y, z) coordinates of the atom.

    Returns
    -------
    modifiers_triple: t.Tuple[float, float, float]
        Triple containing modifiers for the (x, y, z) coordinates
    """
    # Extract discretized coordinates:
    dx, dy, dz = indices
    # Calculate modifiers:
    modifiers_triple = (
        dx - (atom.x / voxel_edge_length + adjust_by),
        dy - (atom.y / voxel_edge_length + adjust_by),
        dz - (atom.z / voxel_edge_length + adjust_by),
    )

    return modifiers_triple


def add_gaussian_at_position(
    main_matrix: np.ndarray,
    secondary_matrix: np.ndarray,
    atom_coord: t.Tuple[int, int, int],
    atom_idx: int,
    atomic_center: t.Tuple[int, int, int] = (1, 1, 1),
    normalize: bool = True,
) -> np.ndarray:
    """
    Adds a 3D array (of a gaussian atom) to a specific coordinate of a frame.

    Parameters
    ----------
    main_matrix: np.ndarray
        Frame 4D array VxVxVxE where V is the n. of voxels and E is the length of
        the encoder.
    secondary_matrix: np.ndarray
        3D matrix containing gaussian atom. Usually 3x3x3
    atom_coord: t.Tuple[int, int, int]
        Coordinates of the atom in voxel numbers
    atom_idx: int
        Denotes the atom position in the encoder (eg. C = 0)
    atomic_center: t.Tuple[int, int, int]
        Center of the atom within the Gaussian representation

    Returns
    -------
    density_frame: np.ndarray
        Frame 4D array with gaussian atom added into it.
    """

    # Copy the main matrix:
    density_frame = main_matrix

    # Remember in a 4D matrix, our 4th dimension is the length of the atom_encoder.
    # This means that to obtain a 3D frame we can just select array[:,:,:, ATOM_NUMBER]
    # Select 3D frame of the atom to be added:
    empty_frame_voxels = np.zeros(
        (density_frame[:, :, :, atom_idx].shape), dtype=np.float16
    )
    # Slice the atom density matrix:
    # This is necessary in case we are at the edge of the frame in which case we
    # need to cut the bits that will not be added.
    density_matrix_slice = secondary_matrix[
        max(atomic_center[0] - atom_coord[0], 0) : atomic_center[0]  # min y
        - atom_coord[0]
        + empty_frame_voxels.shape[0],  # max y
        max(atomic_center[1] - atom_coord[1], 0) : atomic_center[1]  # min x
        - atom_coord[1]
        + empty_frame_voxels.shape[1],  # max x
        max(atomic_center[2] - atom_coord[2], 0) : atomic_center[2]  # min z
        - atom_coord[2]
        + empty_frame_voxels.shape[2],  # max z
    ]
    # Normalize local densities by sum of all densities (so that they all sum up to 1):
    if normalize:
        density_matrix_slice /= np.sum(density_matrix_slice)
    # Slice the Frame to select the portion that contains the atom of interest:
    frame_slice = empty_frame_voxels[
        max(atom_coord[0] - int(density_matrix_slice.shape[0] / 2), 0) : max(
            atom_coord[0] - int(density_matrix_slice.shape[0] / 2), 0
        )
        + density_matrix_slice.shape[0],  # max y
        max(atom_coord[1] - int(density_matrix_slice.shape[1] / 2), 0) : max(
            atom_coord[1] - int(density_matrix_slice.shape[1] / 2), 0
        )
        + density_matrix_slice.shape[1],  # max x
        max(atom_coord[2] - int(density_matrix_slice.shape[2] / 2), 0) : max(
            atom_coord[2] - int(density_matrix_slice.shape[2] / 2), 0
        )
        + density_matrix_slice.shape[2],  # max z
    ]
    # Add atom density to the frame:
    frame_slice += density_matrix_slice
    # Add to original array:
    density_frame[:, :, :, atom_idx] += empty_frame_voxels

    return density_frame


def charge_polar_property(res: Residue, codec: Codec):
    if "P" in codec.atomic_labels:
        if res.mol_letter in standard_amino_acids.keys():
            res_property = -1 if polarity_Zimmerman[res.mol_letter] < 20 else 1
        else:
            res_property = 0
    elif "Q" in codec.atomic_labels:
        res_property = residue_charge[res.mol_letter]
    else:
        res_property = 0
    return res_property


def create_residue_frame(
    residue: Residue,
    frame_edge_length: float,
    voxels_per_side: int,
    # TODO: this argument is not used, but I need to figure out if it should be used!
    # encode_cb: bool,
    codec: Codec,
    voxels_as_gaussian: bool = False,
) -> np.ndarray:
    """Creates a discrete representation of a volume of space around a residue.

    Notes
    -----
    We use the term "frame" to refer to a cube of space around a residue.

    Parameters
    ----------
    residue: ampal.Residue
        The residue to be converted to a frame.
    frame_edge_length: float
        The length of the edges of the frame.
    voxels_per_side: int
        The number of voxels per edge that the cube of space will be converted into i.e.
        the final cube will be `voxels_per_side`^3. This must be a odd, positive integer
        so that the CA atom can be placed at the centre of the frame.
    encode_cb: bool
        Whether to encode the Cb at an average position in the frame.
    codec: object
        Codec object with encoding instructions.
    voxels_as_gaussian: bool
        Whether to encode voxels as a gaussian.

    Returns
    -------
    frame: ndarray
        Numpy array containing the discrete representation of a cube of space around the
        residue.

    Raises
    ------
    AssertionError
        Raised if:

        * If any atom does not have an element label.
        * If any residue does not have a three letter `mol_code` i.e. "LYS" etc
        * If any voxel is already occupied
        * If the central voxel in the frame is not carbon as it should the the CA atom
    """
    assert voxels_per_side % 2, "The number of voxels per side should be odd."
    voxel_edge_length = frame_edge_length / voxels_per_side
    if residue.parent:
        if residue.parent.parent:
            assembly = residue.parent.parent
        else:
            raise ValueError(
                "Residue does not have a parent assembly that will be aligned"
            )
    else:
        raise ValueError("Residue does not have a parent Polymer.")
    align_to_residue_plane(residue)

    frame = np.zeros(
        (voxels_per_side, voxels_per_side, voxels_per_side, codec.encoder_length),
    )
    # Change frame type to float if gaussian else use bool:
    frame = frame.astype(np.float16) if voxels_as_gaussian else frame.astype(bool)
    # iterate through all atoms within the frame
    for atom in (
        a
        for a in assembly.get_atoms(ligands=False)
        if within_frame(frame_edge_length, a)
    ):
        # 3d coordinates are converted to relative indices in frame array
        indices = discretize(atom, voxel_edge_length, adjust_by=voxels_per_side // 2)
        ass = atom.parent.parent.parent
        cha = atom.parent.parent
        res = atom.parent
        assert (atom.element != "") or (atom.element != " "), (
            f"Atom element should not be blank:\n"
            f"{atom.chain}:{atom.res_num}:{atom.res_label}"
        )
        assert (res.mol_code != "") or (res.mol_code != " "), (
            f"Residue mol_code should not be blank:\n"
            f"{cha.id}:{res.id}:{atom.res_label}"
        )
        if not voxels_as_gaussian:
            assert not frame[indices][0], (
                f"Voxel should not be occupied: Currently "
                f"{frame[indices]}, "
                f"{ass.id}:{cha.id}:{res.id}:{atom.res_label}"
            )
            # If the voxel is a gaussian, there may be remnants of a nearby atom
            # hence this test would fail
        if not voxels_as_gaussian:
            if not atom.res_label == "CB":
                np.testing.assert_array_equal(
                    frame[indices], np.array([False] * len(frame[indices]), dtype=bool)
                )
        res_property = charge_polar_property(res, codec)
        # Encode atoms:
        if voxels_as_gaussian:
            modifiers_triple = calculate_atom_coord_modifier_within_voxel(
                atom, voxel_edge_length, indices, adjust_by=voxels_per_side // 2
            )
            # Get Gaussian encoding
            gaussian_matrix, atom_idx = Codec.encode_gaussian_atom(
                codec, atom.res_label, modifiers_triple
            )
            gaussian_atom = gaussian_matrix[:, :, :, atom_idx]
            # Add at position:
            frame = add_gaussian_at_position(
                main_matrix=frame,
                secondary_matrix=gaussian_atom,
                atom_coord=indices,
                atom_idx=atom_idx,
            )
            if res_property != 0:
                gaussian_atom = gaussian_matrix[:, :, :, atom_idx] * float(res_property)
                # Add at position:
                frame = add_gaussian_at_position(
                    main_matrix=frame,
                    secondary_matrix=gaussian_atom,
                    atom_coord=indices,
                    atom_idx=5,
                    normalize=False,
                )
        else:
            # Encode atom as voxel:
            frame[indices] = Codec.encode_atom(codec, atom.res_label)
            if (
                "Q" in codec.atomic_labels
                or "P" in codec.atomic_labels
                and res_property != 0
            ):
                frame[indices] = res_property
    centre = voxels_per_side // 2
    # Check whether central atom is C:
    if "CA" in codec.atomic_labels:
        if voxels_as_gaussian:
            np.testing.assert_array_less(frame[centre, centre, centre][3], 1)
            assert (
                0 < frame[centre, centre, centre][3] <= 1
            ), f"The central atom value should be between 0 and 1 but was {frame[centre, centre, centre][4]}"
        else:
            assert (
                frame[centre, centre, centre][3] == 1
            ), f"The central atom should be Carbon, but it is {frame[centre, centre, centre]}."
    else:
        if voxels_as_gaussian:
            np.testing.assert_array_less(frame[centre, centre, centre][0], 1)
            assert (
                0 < frame[centre, centre, centre][0] <= 1
            ), f"The central atom value should be between 0 and 1 but was {frame[centre, centre, centre][0]}"

        else:
            assert (
                frame[centre, centre, centre][0] == 1
            ), f"The central atom should be Carbon, but it is {frame[centre, centre, centre]}."
    return frame


def voxelize_assembly(
    assembly: Assembly,
    atom_filter_fn,
    name,
    chain_filter_list,
    verbosity,
    chain_dict,
    frame_edge_length,
    voxels_per_side,
    encode_cb,
    codec,
    voxels_as_gaussian,
) -> t.Tuple[str, t.Dict[str, ResidueResult]]:
    # For each monomer in the assembly:
    for polymer in assembly:
        if isinstance(polymer, Polypeptide):
            polymer.tag_sidechain_dihedrals()
    # Filters atoms not related to assembly:
    total_atoms = len(list(assembly.get_atoms()))
    for atom in assembly.get_atoms():
        if not atom_filter_fn(atom):
            del atom.parent.atoms[atom.res_label]
            del atom
    if "CB" in codec.atomic_labels:
        if encode_cb:
            for chain in assembly:
                if isinstance(chain, Polypeptide):
                    for residue in chain._monomers:
                        encode_cb_before_voxelization(residue)
    remaining_atoms = len(list(assembly.get_atoms()))
    print(f"{name}: Filtered {total_atoms - remaining_atoms} of {total_atoms} atoms.")
    for chain in assembly:
        if chain_filter_list:
            if chain.id.upper() not in chain_filter_list:
                if verbosity > 0:
                    print(
                        f"{name}:\tIgnoring chain {chain.id}, not in Pieces filter "
                        f"file."
                    )
                continue
        if not isinstance(chain, Polypeptide):
            if verbosity > 0:
                print(f"{name}:\tIgnoring non-polypeptide chain ({chain.id}).")
            continue
        if verbosity > 0:
            print(f"{name}:\tProcessing chain {chain.id}...")
        chain_dict[chain.id] = []
        # Loop through each residue, voxels:
        for residue in chain:
            if isinstance(residue, Residue) and residue.mol_code:
                # Create voxelized frame:
                array = create_residue_frame(
                    residue=residue,
                    frame_edge_length=frame_edge_length,
                    voxels_per_side=voxels_per_side,
                    codec=codec,
                    voxels_as_gaussian=voxels_as_gaussian,
                )
                encoded_residue = encode_residue(residue.mol_code)
                if "rotamers" in list(residue.tags):
                    if any(v is None for v in residue.tags["rotamers"]):
                        rota = "NAN"
                    else:
                        rota = "".join(
                            np.array(residue.tags["rotamers"], dtype=str).tolist()
                        )
                else:
                    rota = "NAN"
                # Save results:
                chain_dict[chain.id].append(
                    ResidueResult(
                        residue_id=str(residue.id),
                        label=residue.mol_code,
                        encoded_residue=encoded_residue,
                        data=array,
                        voxels_as_gaussian=voxels_as_gaussian,
                        rotamers=rota,
                    )
                )
                if verbosity > 1:
                    print(f"{name}:\t\tAdded residue {chain.id}:{residue.id}.")
        if verbosity > 0:
            print(f"{name}:\tFinished processing chain {chain.id}.")

    return (name, chain_dict)


def default_atom_filter(atom: Atom) -> bool:
    """Filters all heavy protein backbone atoms."""
    backbone_atoms = ("N", "CA", "C", "O")
    if atom.element == "H":
        return False
    elif isinstance(atom.parent, Residue) and (atom.res_label in backbone_atoms):
        return True
    else:
        return False


def keep_sidechain_cb_atom_filter(atom: Atom) -> bool:
    """Filters all heavy protein backbone atoms and the Beta Carbon of
    the side-chain."""
    atoms_to_keep = ("N", "CA", "C", "O", "CB")
    if atom.element == "H":
        return False
    elif isinstance(atom.parent, Residue) and (atom.res_label in atoms_to_keep):
        return True
    else:
        return False


# }}}
