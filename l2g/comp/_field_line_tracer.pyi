import numpy as np
import l2g.comp
import l2g.equil
import l2g.external.tlas
import l2g.external.equilibrium_analysis
import l2g.settings

class FieldLineTracer:
    """This is the class that performs FLT on input mesh data.

    Example usage:

    .. code-block:: py

       import l2g.comp

       flt = l2g.comp.FieldLineTracer()

       # Set the parameters and options

       flt.parameters
       flt.options

       # Set mesh data in the form of vertices and triangles
       flt.setTargetData(v, t)
       # Or just on points
       flt.setTargetData(v)

       # Set the equilibrium data
       flt.setEquilibrium(equilibrium)

       flt.applyParameters()
       flt.processMagneticData()

       # Run the FLT
       flt.runFLT()

       # Process the data

       flt.results

       # Get specific field lines
       flt.fl_ids = [1,2,3,...]
       flt.getFL()

    """
    name: str
    parameters: l2g.settings.Parameters
    options: l2g.settings.Options
    tlas_obj: l2g.external.tlas.PyTLAS
    equilibrium: l2g.equil.Equilibrium
    eq: l2g.external.equilibrium_analysis.EQA
    results: l2g.comp.L2GResults
    recalculate_magnetic_data: bool
    fl_results: l2g.comp.L2GFLs
    owl_conlen_data: np.ndarray
    target_vertices: list
    target_triangles: list
    fl_ids: list
    normals: np.ndarray

    def __init__(self) -> None: ...
    # Input function
    def setParameters(self, parameters: l2g.settings.Parameters) -> None: ...
    def setTLASObj(self, tlas_obj: l2g.external.tlas.PyTLAS) -> None: ...
    def setTargetData(self, vertices: np.ndarray, triangles: np.ndarray | None) -> None:
        """Sets the target data.

        If vertices is provided and triangles is None, then the the vertices
        are already the origin points for the fieldlines.

        If vertices is provided and triangles is not None, then the vertices
        are the vertices of the triangles and triangles contains the ids of
        each triangle.

        The arrays can be whatever dimension as this function flattens it for
        further use.

        Additionally into the cpp vectors the actual origin points are stored,
        either the provided points or the barycenters. In cylindrical
        coordinate system!

        Arguments:
            vertices (np.ndarray): An array of points. These could already be
                the target points or are the vertices of the triangluar mesh.
                In meters
            triangles (np.ndarray | None): An optional numpy array of
                triangles. If this is not none, then the target data is
                considered a mesh. Otherwise the vertices are considered the
                target data.
        """
    def setEquilibrium(self, equilibrium: l2g.equil.Equilibrium) -> None:
        """Set the equilibrium data, which is propagated to the eq analyze
        class and the external FLT C++ code.

        Arguments:
            equilibrium (l2g.equil.Equilibrium): Contains equilibrium magnetic
                data (2D axisymmetric)
        """
    # Pre-run functions
    def applyParameters(self) -> None:
        """Propagates the parameters to the external FLT C++ code. Run this
        before running any FLT run functions.
        """
    def evaluateEq(self) -> None:
        """Runs the equilibrium evaluations functions and prints some basic
        info.
        """
    def flipFieldLineDirections(self) -> None: ...
    def processMagneticData(self) -> None:
        """Here the magnetic data is processed. This function is called
        automatically before any fieldline tracing or fieldline obtaining
        function is called. But in any case if you need the data on the
        target points (either points in space or barycenter of triangular
        mesh) you can call this manually.
        """
    def alignGeometryWithLCFS(self) -> None:
        """In most cases the wall silhouette used in generating the equilibrium
        do not coincide with the geometry used in FLT. Therefore in cases, such
        as limiter cases, where every millimeter counts, we require to align
        the input geometry, so that the equilibrium LCFS is limiting also on
        the geometry.

        !!! note

            In order to use it, the diagnostic for the equilibrium must have a
            the points of the LCFS or boundary contour prepared so that there
            are points to compare.

        The LCFS contour is clipped to the closest element on the mesh.
        Afterwards the distance is set as the displacement for the radial and
        vertical direction. If there are any original displacements, let's say
        vertical already applied, this is already taken into account since the
        LCFS points are following the shift, therefore the underlying C++ code
        will receive the full correct displacement.
        """
    # Run functions
    def getFL(self) -> None:
        """Get fieldlines points. ONLY ON MESH DATA!
        """
    def getFLOnPoint(self, r: float, z: float, phi: float) -> list:
        """Obtain FL points that goes through the input parameters R, Z, phi.
        """
    def runFLT(self) -> None: ...

    # Post process for applying heat load mapping functions
    def calculateDrsep(self) -> None:
        """From the evaluated Flux data and the equilibrium data evaluate the
        radial distance along the midplane for the input target mesh data.
        """
    def applyShadowMask(self, array: np.ndarray) -> np.ndarray:
        """This function applies the shadow mask to an input array.

        ONLY USED WITH MESH RESULTS!!!

        Several criteria are used for determining if a certain area is wetted:

         * Connection length of a field line:
           if conlen > cutoff = True else False
         * Geometries which mark fieldlines as shadowed. This is mainly used in
           regions where the FLs escape the chamber and are not captured by
           anything.

        Arguments:
            array (np.ndarray): A 1D array with N_cells elements on which the
                mask is applied

        Returns:
            masked_array (np.ndarray): A copy of the input array with the mask.

        """
    # Debug
    def debug_getCPoints(self) -> list: ...