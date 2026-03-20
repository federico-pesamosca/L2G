import l2g.external.tlas
import numpy as np

class PyFLT:
    def __init__(self) -> None: ...
    def setPoloidalMagneticFlux(self, r: np.ndarray, z: np.ndarray, psi: np.ndarray) -> None:
        """Sets the poloidal magnetic data.

        The arrays r and z of size NDIMR, NDIMZ defines the [R, Z] domain.

        The psi array is provided as 1D, row oriented and the size must be
        equal to NDIMR * NDIMZ.
        Arguments:
            r (double[:]): 1D vector of radial points. [m]
            z (double[:]): 1D vector of vertical points. [m]
            psi (double[:]): 1D vector of poloidal flux values.
                             [Webb/rad]
        """
    def setVacuumFPOL(self, fpol: float) -> None:
        """Sets the poloidal current function or FPol=Bt * R value in the
        vacuum. This value is used for calculating the toroidal component of
        the magnetic field.

        Arguments:
            fpol (double): Poloidal current function or FPol value in vacuum.
                           In m T (meter Tesla).
        """
    def getVacuumFPOL(self) -> float:
        """Returns the FPol in Vacuum.
        Returns:
            vacuumFpol (double): FPol in Vacuum. In m T.
        """
    def setShift(self, rmove: float, zmove: float) -> None:
        """Sets the radial and vertical shift of the plasma.

        Arguments:
            rmove (double): Radial shift in meters.
            zmove (double): Vertical shift in meters.
        """
    def setAbsError(self, abserr: float) -> None:
        """Sets the absolute error. Less than 1. Default 1e-5

        Arguments:
            abserr (double): Absolute error used by RKF45.
        """
    def setRelError(self, relerr: float) -> None:
        """Sets the relative error. Less than 1. Default 1e-5

        Arguments:
            abserr (double): relative error used by RKF45.
        """
    def setDesiredStep(self, step: float) -> None:
        """Sets the parametric time or toroidal angle end and the resolution
        at which we wish gather points or to check for intersections of a FL.

        Arguments:
            step (double): Toroidal angle step at which we wish to gather
                           points. Used both for obtaining FL points and for
                           FLT intersection.
        """
    def setMaximumFieldLength(self, max_conlen: float) -> None:
        """Sets the maximum fieldline length to which we follow a FL.

        Arguments:
            max_conlen (double): Maximum fieldline length in meters.
        """
    def setSelfIntersectionAvoidanceLength(self, value: float) -> None:
        """Sets the initial length at which we avoid intersection tests. For
        example 0.005 (meters), this means that at the first 5 millimeters, do
        not check for intersection tests.

        Arguments:
            value (double): Initial length at which intersection tests are
                            ignored. In meters.
        """
    def runFLT(self) -> None:
        """Calls the runFLT function which starts the FLT.
        """

    def getFL(self, r: float, z: float, phi: float) -> list[float]:
        """Get fieldline going through point (r, z, phi). Output is a list
        containing two parts of the fieldline:
            [[r1, z1, phi1, r2, z2, phi2],
             [r1, z1, phi1, rn+1, zn+1, phin+1, ...]]

        Arguments:
            r (float): radial position. In meters
            z (float): vertical position. In meters
            phi (float): toroidal angle position. In radians

        Returns:
            out(list[list[float]]): List that contains two lists. One list
                contains the fieldline that starts with the up direction and
                the other contains the part that starts with the down direction
        """
    def getBCart(self, r: float, z: float, phi: float) -> list[float]:
        """Get the B vector at point (r, z, phi) in Cartesian coordinate
        system.

        Arguments:
            r (float): Radial position
            z (float): Vertical position
            phi (float): Angle in toroidal direction

        Returns:
            out (list[float]): (Bx, By, Bz)
        """
    def getBCyln(self, r: float, z: float) -> list[float]:
        """Get the B vector at point (r, z) in Cylindrical coordinate system.

        Arguments:
            r (double): Radial position
            z (double): Vertical position

        Returns:
            out (list[float]): (Bpol, Btor)
        """
    def getPoloidalFlux(self, r: float, z: float) -> float:
        """Gets the polodidal magnetic flux value at point (r, z).

        Arguments:
            r (double): Radial position
            z (double): Vertical position

        Returns:
            flux (double): Poloidal magnetic flux. In Webb/rad.
        """
    def applyRT(self, obj: l2g.external.tlas.PyTLAS) -> None:
        """This function is used to bind an TLAS C++ object to the FLT C++
        object. Only pointer is used, so essentially we could switch different
        TLAS objects.

        Arguments:
            tlas_obj (l2g.external.tlas.PyTLAS): TLAS object.
        """