# The following import line is used only for type hints. Remove it and the
# used types to avoid importing the main module.
import numpy as np
import logging
import os
from types import ModuleType

log = logging.getLogger(__name__)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from l2g.comp import L2GResults, L2GResultsHLM, L2GFLs

def supportedFileExts() -> list[str]:
    return ["med", "vtk", "vtu"]

def rotatePointsAroundAxis(points: np.ndarray, p1: np.ndarray,
                           p2: np.ndarray, theta: float) -> np.ndarray:
    """Rotate the argument points around the axis :math:`|p2 - p1|` for the
    angle theta.

    Arguments:
        points (npt.ndarray): Array of points with the shape ((N_points, 3)).
        p1 (npt.ndarray): Start point of the axis
        p2 (npt.ndarray): End point of the axis
        theta (float): angle in degrees.

    Returns:
        rotated_points (np.ndarray)
    """

    # Copy the points and translate it to p1.
    out_points = np.zeros(points.shape)
    out_points[:] = points - p1

    # Get the normalized diff vector.
    diff_p = p2 - p1
    u = diff_p / np.linalg.norm(diff_p)

    # Construct the rotational matrix
    R = np.zeros((3, 3))

    cos_th = np.cos(np.deg2rad(theta))
    sin_th = np.sin(np.deg2rad(theta))
    moin_cos_th = 1 - cos_th
    ux = u[0]
    uy = u[1]
    uz = u[2]
    R[0, 0] = cos_th + ux * ux * moin_cos_th
    R[1, 0] = uy * ux * moin_cos_th + uz * sin_th
    R[2, 0] = uz * ux * moin_cos_th - uy * sin_th

    R[0, 1] = ux * uy * moin_cos_th - uz * sin_th
    R[1, 1] = cos_th + uy * uy * moin_cos_th
    R[2, 1] = uz * uy * moin_cos_th + ux * sin_th

    R[0, 2] = ux * uz * moin_cos_th + uy * sin_th
    R[1, 2] = uy * uz * moin_cos_th - ux * sin_th
    R[2, 2] = cos_th + uz * uz * moin_cos_th

    # Rotate the points
    # out_points = R@out_points.T
    out = (R@out_points.T).T

    # We need to fix the stride. Best way is to copy all values...
    # No idea otherwise how to fix the array stride.
    # It can be done with "np.ascontigous" array.
    out_points[:] = out
    # out_points = out_points.T

    # Re-translate it to correct place.
    # out_points += p1
    return np.array(out_points + p1, dtype=np.float32)

def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return an array of areas of triangles.

    Arguments:
        vertices (np.ndarray): Array of points
        triangles (np.ndarray): Array of triangles
    """
    p1 = vertices[triangles[:, 0]]
    p2 = vertices[triangles[:, 1]]
    p3 = vertices[triangles[:, 2]]

    p12 = p2 - p1
    p13 = p3 - p1
    cross = np.cross(p12, p13)
    return 0.5 * np.linalg.norm(cross, axis=1)

def triangle_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return an array of normals of triangles

    Arguments:
        vertices (np.ndarray): Array of points of shape (N, 3)
        triangles (np.ndarray): Array of triangles of shape (N, 3)

    Returns:
        normals (np.ndarray): 2D array (N, 3) of normals.
    """
    p1 = vertices[triangles[:, 0]]
    p2 = vertices[triangles[:, 1]]
    p3 = vertices[triangles[:, 2]]

    p12 = p2 - p1
    p13 = p3 - p1
    cross = np.cross(p12, p13)
    # return cross / np.linalg.norm(cross, axis=1) #change this with the following line
    return cross / np.linalg.norm(cross, axis=1)[:, np.newaxis]

class WrongFileExtensionException(Exception):
    pass

class FileDoesNotExistException(Exception):
    pass

class FieldNotInMesh(Exception):
    pass

class GroupNotInMesh(Exception):
    pass

class NoMeshDataLoaded(Exception):
    pass

def integrate_quantity(mesh: 'Mesh', field_name: str, area_density: bool=True,
                       opt_groups: list[str] | None = None) -> float | dict[str, float]:
    """Integrate by time and return the value of a quantity on the provided
    Mesh object. Additionally if a list of groups (sub-meshes) are provided,
    then the output is a dictionary.

    Arguments:
        mesh (Mesh): Mesh object.
        field_name (str): Quantity to integrate.
        area_density (bool): If true, treat quantity as area density, or the
            units are per unit of area.
        opt_groups (list[str] | None): Optional name of subgroups.

    Returns:
        result (float | dict): Returns the integrated value or a dictionary if
            a list of names is provided.
    """

    if field_name not in mesh.getAllFieldNames():
        raise FieldNotInMesh(f"Field {field_name} not in mesh")
    iterations = mesh.getAllFieldIterations(field_name)

    if not opt_groups is None:
        all_groups = mesh.getAllGroups()
        for name in opt_groups:
            if name not in all_groups:
                raise GroupNotInMesh

    if len(iterations) < 2:
        return 0 if opt_groups is None else {_: 0 for _ in opt_groups}

    # Integrate the quantity
    mesh.setIndex(iterations[0][0])
    pf = mesh.getField(field_name)
    pt = iterations[0][1]

    q = np.zeros(pf.shape, pf.dtype)

    for i, t in iterations[1:]:
        mesh.setIndex(i)
        f = mesh.getField(field_name)

        q += 0.5 * (pf + f) * (t - pt)

        pf = f
        pt = t

    # Multiply by area if area density.
    if area_density:
        q *= mesh.getCellMeasurements()

    if opt_groups is None:
        return float(np.sum(q))

    out = {}
    for name in opt_groups:
        array = mesh.getGroupArray(name)
        out[name] = float(np.sum(q[array]))
    return out

class Mesh():
    """This is a convenience class for reading/writing 2D surface mesh made of
    triangles and with time-dependent data on it. Primarily it supports the
    MED format (MED-file, MEDCoupling) but it is intended to be an easy to use
    interface with different I/O backends behind.

    Important that its main use is for reading/writing to Unstructured meshes
    made of triangles. Any other cell type is ignored.

    .. code-block:: python

       import l2g.mesh
       m = l2g.mesh.Mesh("/path/to/file.med")

       # Get mesh data
       vertices, triangles = m.getMeshData()
       # Get measurements of cells (Useful for integrating flux quantities)
       measurements = m.getMeasurements()

       # See if the field exists in the file
       ok = m.doesItContainField("field_name")

       # Get all field iterations available
       iterations = m.getAllFieldIterations("field_name")

       # Get field at index n
       n = 5 # Or some integer value
       m.setIndex(n)
       field_numpy_array = m.getField("field_name")

       # Add a field
       import numpy as np
       custom_array = np.random.random(vertices.shape[0], np.float)

       # Specify the time index of the field. Important before adding fields.
       m.setIndex(0)

       m.addField("custom_field_name", custom_array)

       # Maybe we have a vector field.
       vector_array = np.random.random(vertices.shape, np.float)
       # Let's also add information on each component of the vector
       infoOnComponents = ["x", "y", "z"]
       m.addField("custom_vector_field", vector_array, infoOnComponents)

       # You can now add fields at different time indexes if you want to write
       # it in bulk.

       # Finally write the fields.
       m.writeFields()

    """
    def __init__(self, file_path: str=""):
        self.mesh_name: str = "mesh"

        self.vertices: np.ndarray = np.array([])
        self.triangles: np.ndarray = np.array([])

        # Field index
        self.index: int = 0
        self.time: float = 0.0
        self.number_of_time_steps: int = 1

        # Fields
        self.arrays: dict[str, dict[int, np.ndarray]] = {}

        # Writing data
        self.info_on_components: dict[str, dict[int, list[str]]] = {}
        self.times: dict[str, dict[int, float]] = {}

        # MODULE pointer
        self.backend: ModuleType = None
        self.backend_name: str = 'UNKNOWN'
        if file_path:
            self.file_path: str = file_path
            self.openFile(self.file_path)
        else:
            self.file_path = ""

    def openFile(self, file_path: str):
        """Open the file by creating the appropriate backend objects necessary
        for I/O operations.

        If the function succeeds it returns nothing. Otherwise if the file
        extension does not match or the file does not exist, it raises either
        WrongFileExtensionException or FileDoesNotExistException respectively.
        """
        # Get the extension
        ext = os.path.splitext(file_path)[-1].lower()
        if ext.startswith("."):
            ext = ext[1:]

        if ext not in supportedFileExts():
            log.error(f"File {file_path} does not have a supported file ext.")
            log.info(f"Supported exts: {' '.join(supportedFileExts())}")
            self.backend_name = ""
            self.backend = None
            raise WrongFileExtensionException(f"{ext} not supported.")

        if not os.path.exists(file_path):
            log.error(f"File {file_path} does not exist!")
            raise FileDoesNotExistException(f"File {file_path} does not exist")

        self.file_path = file_path

        if ext == "med":
            from . import medio
            self.backend = medio
            self.backend_name = "MED"
        elif ext.startswith('vt'):
            from . import _vtk
            self.backend = _vtk
            self.backend_name = "VTK"
        else:
            self.backend_name = "UNKNOWN"
            self.backend = None

    def getMeshData(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the vertices and triangles of the mesh. Only the triangle cells
        are obtained as it is assumed that this class works with surface type
        meshes.

        Returns:
            vertices (np.ndarray): 2D array of points with the shape
                (N_vertices, 3).
            triangles (np.ndarray): 2D array of triangles with the shape
                (N_triangles, 3).

        """
        if self.vertices.size == 0:
            self.readMeshData()

        if self.backend is None:
            raise NoMeshDataLoaded

        return self.vertices, self.triangles

    def getCellMeasurements(self) -> np.ndarray:
        """Returns a 1D array containing the measurements of the cells. The
        index of a value corresponds to the index of a cell.

        Measurements:
         - 2D cell = area.

        Returns:
            measurements (np.ndarray): 1D array of floats.

        """
        if self.backend is None:
            raise NoMeshDataLoaded

        if self.vertices.size == 0:
            self.readMeshData()

        return triangle_areas(self.vertices, self.triangles)

    def getNormals(self) -> np.ndarray:
        """Returns a 2D array of shape (N, 3) of cell normals.

        Returns:
            normals (np.ndarray): 2D array of normals.
        """
        if self.backend is None:
            raise NoMeshDataLoaded

        if self.vertices.size == 0:
            self.readMeshData()

        return triangle_normals(self.vertices, self.triangles)

    def readMeshData(self) -> None:
        """Function that calls the backends to fetch the mesh data of the file
        and storing it.
        """
        if self.backend is None:
            raise NoMeshDataLoaded

        if self.vertices.size == 0:
            v, t = self.backend.getMeshData(self.file_path)
            self.vertices = v
            self.triangles = t

    def setIndex(self, index: int) -> None:
        """Sets the index for reading/writing fields. This value is used when
        writing functions with the :py:meth:`l2g.mesh.Mesh.writeFields` as it
        holds information at which index to write the fields.

        In the :py:meth:`l2g.mesh.Mesh.getField` this index is used to obtain
        the field data at that index.
        """
        self.index = index

    def setTime(self, time: int | float) -> None:
        """Sets the time information. This value is used only as description,
        as in, when you visualize the results in software like ParaView, this
        field will be describing the time of the time step (index) of the
        data.

        This information is used when using the function
        :py:meth:`l2g.mesh.Mesh.writeFields`.
        """
        self.time = time

    def setNumberOfTimeSteps(self, number_of_time_steps: int) -> None:
        """Define how many time steps are there in the file. Use this when
        writing data to a file. Some writers (VTK) requires this information
        before writing the data.

        Arguments:
            number_of_time_steps (int): Number of time steps.
        """
        if number_of_time_steps <= 0:
            return
        self.number_of_time_steps = number_of_time_steps

    def getNumberOfTimeSteps(self) -> int:
        """Get number of time steps in a FLT mesh file. It will look for the
        most common FLT field, which is conlen.

        Returns:
            number_of_fields(int): Number of time fields in the file.
        """
        if self.backend is None:
            return 0
        flt_field = "conlen"
        return self.backend.getNumberOfTimeSteps(self.file_path, flt_field)

    def doesItContainField(self, field_name: str) -> bool:
        """Checks if the mesh object contains the field with the name.

        Returns:
            ok (bool): True if mesh contains field with this name
        """
        if self.backend:
            return self.backend.checkIfFieldExists(self.file_path, field_name)
        else:
            return False

    def getAllFieldNames(self) -> list[str]:
        """Get the name of all quantities in the mesh files.
        """

        if self.backend:
            return self.backend.getAllFieldNames(self.file_path)
        return []

    def getAllFieldIterations(self, field_name: str) -> list[tuple[int, float]]:
        """Returns all indexes and times of the field inside the file.

        Returns:
            out (List[Tuple[int, float]]): A list containing pairs of index
                and times (i.e., [[0, 0.0], [1, 0.5]].
        """

        out: list[tuple[int, float]] = []

        if self.backend is None:
            return []

        if self.doesItContainField(field_name):
            out = self.backend.getAllFieldIterations(self.file_path, field_name)
            out = [(_[0], _[1]) for _ in out]

        return out

    def getField(self, field_name: str,
                 index: None | int =None) -> np.ndarray:
        """Returns a field of the the opened file. Index is always provided,
        where it signifies the time step of the field.

        Returns:
            field (np.ndarray): A numpy array of the field.
        """
        if index is None:
            index = self.index
        if self.backend:
            field = self.backend.getField(self.file_path, field_name, index)
            if field is None:
                msg = f"There is no {field_name} at index {index} in {self.file_path}"
                log.error(msg)
                raise FieldNotInMesh
            return field
        else:
            raise WrongFileExtensionException

    def doesItContainGroup(self, group_name: str) -> bool:
        if self.backend:
            return self.backend.doesItContainGroup(self.file_path, group_name)
        return False

    def getGroupArray(self, group_name: str) -> np.ndarray:
        if self.backend:
            return self.backend.getGroupArr(self.file_path, group_name)
        return np.array([])

    def getAllGroups(self) -> list[str]:
        """Returns a list of all defined groups inside the MED file.
        """
        if self.backend:
            return self.backend.getAllGroups(self.file_path)
        return []

    def writeMeshTo(self, file_path: str) -> 'Mesh':
        """In this case we wish to copy the mesh of a file to a new location.
        Same file paths are *not* allowed, i.e., over-writing mesh in the same
        file.

        The function returns a new Mesh object that has a copy of only mesh
        data.

        Arguments:
            file_path (str): Path to new file.

        Returns:
            mesh_obj (Mesh): Mesh object with reference to the new file path.

        """
        if file_path == self.file_path:
            return self

        if not self.vertices.size:
            # Try getting the data
            self.readMeshData()
            if not self.vertices.size:
                raise NoMeshDataLoaded()

        # Figure out the correct backend.
        ext = os.path.splitext(file_path)[-1].lower()
        if ext.startswith("."):
            ext = ext[1:]
        backend = None
        if ext == "med":
            from . import medio
            backend = medio
        elif ext == "vtu":
            from . import _vtk
            backend = _vtk

        if backend is None:
            raise NoMeshDataLoaded()
        backend.writeMesh(file_path, self.vertices, self.triangles,
                          self.mesh_name)

        # Make a copy
        new_obj = Mesh()
        new_obj.mesh_name = self.mesh_name
        # Copy only mesh
        new_obj.vertices = np.copy(self.vertices)
        new_obj.triangles = np.copy(self.triangles)
        new_obj.openFile(file_path)
        return new_obj

    def addField(self, array_name: str, array: np.ndarray,
                 info_on_components: list[str]=[""]):
        """Add field to be written to a file. The input array is actually a
        numpy array, but when we are talking on saving it to files, where the
        array corresponds to an unstructured grid made of triangles, it is
        named a field.

        Arguments:
            array_name (str): Name of the field. It is named array as the data
                is an array, it becomes a field when written in the file.
            array (np.ndarray): Numpy array. Can be multi-dimensional, i.e.,
                vector data.
            info_on_components (list): When we have vector data it is necessary to also
                provides the info_on_components of each component of the vector data.
        """
        if len(info_on_components) == 0:
            log.error('Field components cannot be an empty list. At the ' +
                      'bare minimum it has to be [""]')

        if array_name not in self.arrays:
            self.arrays[array_name] = {}

        if array_name not in self.info_on_components:
            self.info_on_components[array_name] = {}

        if array_name not in self.times:
            self.times[array_name] = {}
        index = self.index
        self.arrays[array_name][index] = array
        self.info_on_components[array_name][index] = info_on_components
        self.times[array_name][index] = self.time

    def writeFields(self) -> None:
        """Write the fields to the mesh file, which were added with the function
        :py:meth:`l2g.mesh.Mesh.addField`.
        """

        if self.backend is None:
            log.error("No valid backend.")
            return

        if self.backend_name == "MED":
            from . import medio
            if self.file_path in medio.__medtr3reader__:
                reader = medio.__medtr3reader__[self.file_path]
            else:
                reader = medio.MEDTR3Reader(self.file_path)

            mesh_name = reader.mesh_names[0]

            writer = medio.MEDTR3Writer(mesh_name)

            for array_name in self.arrays:
                register_field = True
                for index in self.arrays[array_name]:
                    array = self.arrays[array_name][index]
                    time = self.times[array_name][index]
                    components = self.info_on_components[array_name][index]
                    n_components = len(components)
                    if register_field:
                        writer.registerField(array_name, components)
                        register_field = False

                    # Check if array has the correct shape.
                    if n_components > 1 and array.ndim < 2:
                        # Reshape the array correctly.
                        if array.size % n_components != 0:
                            raise Exception(f'Field {array_name}, size {array.size} is not divisible by {n_components}')

                        array = array.reshape((-1, n_components))

                    writer.addFieldData(array_name, index, time, array)
            writer.writeField(self.file_path)

            reader.traverse() # Update the reader.

        elif self.backend_name == "VTK":
            # Haven't found a way to append array data to an XML Unstructured
            # Grid data, though if I wrote my own XML writer it would be
            # easier. In this case the VTU, or the XML Unstructured Grid file
            # is supported as it can actually contain multiple time steps in a
            # single file. Other formats requires additional XML, or files
            # to be generated.

            from . import _vtk

            # Manually do the thing as it is quite complicated.
            writer = _vtk.getWriter(self.file_path)
            vtk_orig_obj = _vtk.getVtkObj(self.file_path, self.vertices,
                                                 self.triangles)

            if self.index == 0:
                # We are starting!
                writer.SetFileName(self.file_path)
                writer.SetNumberOfTimeSteps(self.number_of_time_steps)
                writer.Start()

            vtk_obj = _vtk.copyVtkObj(vtk_orig_obj)

            # The problematic code is following, where we have to dump the
            # arrays to the VTK object.
            for array_name in self.arrays:
                _vtk.addArrayToVtkObj(vtk_obj, array_name,
                    self.arrays[array_name][self.index])

            # Now write
            writer.SetInputData(vtk_obj)
            writer.WriteNextTime(self.time)

            if self.index == self.number_of_time_steps - 1:
                # Stop the writing when we reach the number of time steps to
                # write. Note that the setNumberOfTimesteps function must be
                # called in order for this to work.
                writer.Stop()

        # Clear the data arrays.
        # Fields
        self.arrays = {}

        # Writing data
        self.info_on_components = {}
        self.times = {}

        return

    def translateMesh(self, vector: np.ndarray) -> None:
        """Translates the vertices with the provided vector.

        Arguments:
            vector (np.ndarray): Direction vector in which the mesh points are
                translated.
        """
        if self.file_path is None:
            log.error("Could not read mesh data as no MED file is loaded")
            return None

        if self.vertices.size == 0:
            self.readMeshData()
        self.vertices += vector

    def rotateMesh(self, p1: np.ndarray, p2: np.ndarray, angle: float) -> 'Mesh':
        """Rotates the the mesh data around the provided axes and angle. This
        rotation is not "free form", the rotational axis is set to the
        geometry's center of mass.

        It creates a new Mesh, since it creates a copy in any case and it
        is sometimes useful if we retain the original data.

        Arguments:
            p1 (np.ndarray): First point of the rotational vector axis
            p2 (np.ndarray): Second point of the rotational vector axis
            angle (float): Angle of rotationa

        Returns:
            mesh (Mesh): New mesh object with rotated vertices.
        """

        if self.file_path is None:
            log.error("Could not read mesh data as no MED file is loaded")
            raise NoMeshDataLoaded()

        if self.vertices.size == 0:
            self.readMeshData()

        out = Mesh()
        vertices = self.vertices.copy()

        # Now rotate
        new_vertices = rotatePointsAroundAxis(vertices, p1, p2, angle)
        out.vertices = new_vertices
        out.triangles = self.triangles.copy()
        return out

def load_flt_results_from_mesh(mesh_results: 'L2GResults', mesh_object: Mesh):
    """Loads required fields from the Mesh objects to the mesh results.

    It is necessary to set the index of the fields *before* calling this
    function in order to get the correct data.

    .. code-block:: python

       mesh_object.setIndex(n)
       l2g.mesh.load_flt_results_from_mesh(mesh_results, mesh_object)

    Arguments:
        mesh_results (L2GResults): Object which will have data loaded on
        mesh_object (Mesh): Object from where to take the data
    """
    if mesh_object.backend is None:
        log.error("Mesh file is not open to load data from")
        return

    mesh_results.drsep = mesh_object.getField('drsep') * 1e-3 # Convert to m
    mesh_results.conlen = mesh_object.getField('conlen') * 1e-3
    mesh_results.flux = mesh_object.getField('flux')

    # Necessary for the mask.
    mesh_results.geom_hit_ids = mesh_object.getField("geom_hit_ids")

    # Necessary for FL
    # Necessary to flatten!
    mesh_results.BVec = mesh_object.getField("BVec").flatten()
    mesh_results.Bdot = mesh_object.getField("Bdot")
    # Necessary to flatten!
    # mesh_results.baryCent = mesh_object.getField("baryCent").flatten()
    mesh_results.empty = False

def dump_flt_results_to_mesh(mesh_results: 'L2GResults', mesh_object: Mesh):
    """Save the results data in mesh_results to the mesh_object and write it to
    a file.

    It is necessary to set the index of the fields *before* calling this
    function in order to write at the correct index and time step.

    .. code-block:: python

       mesh_object.setIndex(n)
       mesh_object.setTime(time)
       l2g.mesh.dump_flt_results_to_mesh(mesh_results, mesh_object)

    Arguments:
        mesh_results ('L2GResults'): Class object holding data
        mesh_object (Mesh): Mesh object instance
    """

    for key in mesh_results.arrays_to_dump:
        array = getattr(mesh_results, key)
        if array is None:
            continue

        if key == "angle":
            # Transform the angle
            array = np.rad2deg(array)
            array = np.where(array > 90.0, array - 90.0, 90.0 - array)

        if key in ["drsep", "drsep2", "conlen"]:
            # Scale to mm from m
            array = array * 1000

        infoOnComponent = []
        if key == 'BVec':
            # Write the Magnetic field vector properly. That is a
            # 3 component array in Cartesian!
            infoOnComponent = ['x', 'y', 'z']
        elif key == 'normals':
            infoOnComponent = ['x', 'y', 'z']
        elif key == 'BVecCyln':
            # Write the Magnetic field poloidal and toroidal component
            # properly. A 2 component array in Cylindrical
            infoOnComponent = ['Pol', 'Tor']
        elif key == 'baryCent':
            # Again, write the baryCent properly. In this case the
            # infoOnComponent will basically tell the user that the vector
            # is in Cylindrical system.
            infoOnComponent=['r', 'z', 'phi']

        mesh_object.addField(array=array, array_name=key,
                             info_on_components=infoOnComponent)

    mesh_object.writeFields()

def dump_hlm_results_to_mesh(hlm_results: 'L2GResultsHLM', mesh_object: Mesh,
                             hlm_type: str):
    """Dumps flt_mesh results to mesh object.

    hlm_results is L2GResultsHLM

    """
    # Classical arrays.
    mesh_object.addField(array=hlm_results.flux_expansion,
                         array_name="Total_flux expansion")
    mesh_object.addField(array=hlm_results.q_inc,
        array_name=r"$q_{\perp}\;\;[\frac{W}{m^2}]$")
    mesh_object.addField(array=hlm_results.q_par,
        array_name=r"$q_{\parallel}\;\;[\frac{W}{m^2}]$")

    if hlm_type == "elm":
        # Writing additional arrays
        # Take arrays from additional_arrays variable.
        mesh_object.addField(
            array=hlm_results.additional_arrays[0],
            array_name=r"$q_{\parallel}\;\;[\frac{W}{m^2}]$ ELM")
        mesh_object.addField(
            array=hlm_results.additional_arrays[1],
            array_name=r"$q_{\parallel}\;\;[\frac{W}{m^2}]$ inter-ELM")
        mesh_object.addField(
            array=hlm_results.additional_arrays[2],
            array_name=r"$T_e$ [eV]")
        mesh_object.addField(
            array=hlm_results.additional_arrays[3],
            array_name=r"$T_i$ [eV]")
        mesh_object.addField(
            array=hlm_results.additional_arrays[4],
            array_name=r"$q_{\perp}\;\;[\frac{W}{m^2}]$ ELM")
        mesh_object.addField(
            array=hlm_results.additional_arrays[5],
            array_name=r"$q_{\perp}\;\;[\frac{W}{m^2}]$ inter-ELM")

    mesh_object.writeFields()

def save_fls_to_vtk(data, file_path):
    """Helper function that saves field lines points or polylines to a VTK
    file. In this case ``vtkGenericDataObjectWriter`` is used.

    This should be used to save :class:`L2GFLs` to a VTK file.
    """
    from vtkmodules.vtkIOLegacy import vtkGenericDataObjectWriter
    writer = vtkGenericDataObjectWriter()
    writer.SetInputData(data)
    writer.SetFileName(file_path)
    writer.Write()

def save_mesh_to_vtk(data, file_path):
    """Helper function that saves field lines points or vtkUnstructuredGrid to
    a VTK file. In this case ``vtkXMLUnstructuredGridWriter`` is used.

    This should be used to save :class:`L2GResults` to a VTK file.
    """
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetInputData(data)
    writer.SetFileName(file_path)
    writer.Write()

def save_results_to_vtk(result_obj: 'L2GFLs | L2GResults',
                       file_path: str) -> None:
    """Helper function to save result objects: :class:`L2GResults`,
    :class:`L2GFLs` to a file.

    In case of :class:`L2GResults` the data is saved to an existing MED file,
    so the MED file has to exist.

    In case of :class:`L2GFLs` a new VTK file is created

    """
    from l2g.comp import L2GResults, L2GFLs

    data = result_obj.generate_vtk_object()

    if isinstance(result_obj, L2GResults):
        save_mesh_to_vtk(data, file_path)
    elif isinstance(result_obj, L2GFLs):
        save_fls_to_vtk(data, file_path)
