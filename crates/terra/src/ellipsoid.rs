//! Triaxial ellipsoid - the mathematical core of geodetic calculations.

use glam::{DMat4, DVec2, DVec3};
use outil::EPSILON12;
use zukei::Plane;

use crate::{Cartographic, transforms::east_north_up_to_ecef};

/// Triaxial ellipsoid defined by its three semi-axis radii (in metres).
///
/// Construct via [`Ellipsoid::wgs84`], [`Ellipsoid::unit_sphere`], or
/// [`Ellipsoid::new`]. All geodetic conversions use the ellipsoid's radii
/// so the same algorithms work for WGS84, GRS80, the Moon, Mars, etc.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ellipsoid {
    /// Semi-axis radii (x, y, z) in metres.
    pub radii: DVec3,
    radii_squared: DVec3,
    one_over_radii: DVec3,
    one_over_radii_squared: DVec3,
}

// default ellipsoid is wgs84
impl Default for Ellipsoid {
    fn default() -> Self {
        Self::wgs84()
    }
}

impl Ellipsoid {
    // WGS84 semi-major / semi-minor axes (metres).
    pub const WGS84_A: f64 = 6_378_137.0;
    pub const WGS84_B: f64 = 6_356_752.314_245_179_3;

    /// WGS84 ellipsoid (the standard for GPS / ECEF).
    pub fn wgs84() -> Self {
        Self::new(Self::WGS84_A, Self::WGS84_A, Self::WGS84_B)
    }

    /// Unit sphere (r = 1 in all axes). Useful for tests and normalised math.
    pub fn unit_sphere() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Construct an ellipsoid with explicit semi-axis radii.
    ///
    /// # Panics
    /// Panics if any radius is zero or negative.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        assert!(
            x > 0.0 && y > 0.0 && z > 0.0,
            "all ellipsoid radii must be positive"
        );
        let radii = DVec3::new(x, y, z);
        Self {
            radii,
            radii_squared: radii * radii,
            one_over_radii: DVec3::new(1.0 / x, 1.0 / y, 1.0 / z),
            one_over_radii_squared: DVec3::new(1.0 / (x * x), 1.0 / (y * y), 1.0 / (z * z)),
        }
    }

    /// Convert geodetic coordinates to ECEF Cartesian (metres).
    ///
    /// Uses the standard closed-form formula:
    /// ```text
    /// N = a / sqrt(1 - e^2 sin^2\phi)
    /// X = (N + h) cos\phi cos\lambda
    /// Y = (N + h) cos\phi sin\lambda
    /// Z = (N(1-e^2) + h) sin\phi
    /// ```
    pub fn cartographic_to_ecef(&self, c: Cartographic) -> DVec3 {
        let e2 = self.first_eccentricity_squared();
        let sin_lat = c.latitude.sin();
        let cos_lat = c.latitude.cos();
        let n = self.radii.x / (1.0 - e2 * sin_lat * sin_lat).sqrt();
        DVec3::new(
            (n + c.height) * cos_lat * c.longitude.cos(),
            (n + c.height) * cos_lat * c.longitude.sin(),
            (n * (1.0 - e2) + c.height) * sin_lat,
        )
    }

    /// Convert ECEF Cartesian to geodetic coordinates.
    ///
    /// Equivalent to Cesium's `Ellipsoid.cartesianToCartographic`. Projects
    /// the point onto the ellipsoid surface, reads the surface normal to get
    /// longitude and latitude, then computes height as the signed distance
    /// along that normal.
    ///
    /// Returns `None` if the point is too close to the centre of the ellipsoid
    /// (less than ~31% of minimum radius) where the normal is undefined.
    pub fn ecef_to_cartographic(&self, ecef: DVec3) -> Option<Cartographic> {
        let surface = self.scale_to_geodetic_surface(ecef)?;
        let normal = self.geodetic_surface_normal(surface);
        let h = ecef - surface;
        // sign comes from whether `h` and the input position point the same way
        // (point above surface vs. inside the ellipsoid), magnitude is the
        // Euclidean distance from the surface. This is robust for points
        // below the surface or off the geodetic normal from the surface
        // projection, which the previous `h.dot(normal)` form was not.
        let sign = if h.dot(ecef) >= 0.0 { 1.0 } else { -1.0 };
        let height = sign * h.length();
        // Clamp the normal's z component before `asin`: after `normalize`,
        // rounding can leave |z| infinitesimally above 1.0, which would
        // otherwise produce NaN latitudes for points on the poles.
        let latitude = normal.z.clamp(-1.0, 1.0).asin();
        let longitude = normal.y.atan2(normal.x);
        Some(Cartographic {
            longitude,
            latitude,
            height,
        })
    }

    /// Compute the unit geodetic surface normal at an ECEF position.
    ///
    /// The normal at surface point P is proportional to `P / radii^2` (element-wise)
    #[inline]
    pub fn geodetic_surface_normal(&self, ecef: DVec3) -> DVec3 {
        (ecef * self.one_over_radii_squared).normalize()
    }

    /// Compute the geodetic surface normal directly from cartographic coordinates.
    ///
    /// Equivalent to `(cos\phi cos\lambda, cos\phi sin\lambda, sin\phi)` - cheaper than converting to
    /// ECEF and calling [`geodetic_surface_normal`].
    #[inline]
    pub fn geodetic_surface_normal_at(&self, c: Cartographic) -> DVec3 {
        let cos_lat = c.latitude.cos();
        DVec3::new(
            cos_lat * c.longitude.cos(),
            cos_lat * c.longitude.sin(),
            c.latitude.sin(),
        )
    }

    /// Project an ECEF point onto the ellipsoid surface along the geodetic normal.
    ///
    /// Solves, via Newton-Raphson, for the scalar λ such that:
    /// `\sum{ (P_{i} x oor_{i} / (1 + \lambda x oor^{2}_i))^2} = 1`
    ///
    /// Returns `None` if the input is within `centerToleranceSquared` of the
    /// ellipsoid centre (matching Cesium's behaviour).
    pub fn scale_to_geodetic_surface(&self, ecef: DVec3) -> Option<DVec3> {
        let oor = self.one_over_radii;
        let oor2 = self.one_over_radii_squared;

        // Scaled squared norms.
        let x2 = (ecef.x * oor.x) * (ecef.x * oor.x);
        let y2 = (ecef.y * oor.y) * (ecef.y * oor.y);
        let z2 = (ecef.z * oor.z) * (ecef.z * oor.z);
        let squared_norm = x2 + y2 + z2;

        // Reject points near the centre (surface normal undefined).
        if squared_norm < 0.1 {
            return None;
        }

        // Already on the surface?
        if (squared_norm - 1.0).abs() < EPSILON12 {
            return Some(ecef);
        }

        // Warm-start \u03bb using the gradient at the radial projection of the
        // input onto the ellipsoid. Starting from \u03bb = 0 converges
        // much more slowly for points far from the surface.
        let ratio = (1.0 / squared_norm).sqrt();
        let intersection = ecef * ratio;
        let gradient = DVec3::new(
            intersection.x * oor2.x * 2.0,
            intersection.y * oor2.y * 2.0,
            intersection.z * oor2.z * 2.0,
        );
        let mut lambda = ((1.0 - ratio) * ecef.length()) / (0.5 * gradient.length());

        // Newton-Raphson: find \u03bb such that f(\u03bb) = 0 where
        //   f(\u03bb) = \u03a3 (x\u1d62 / (1 + \u03bb\u00b7oor\u00b2\u1d62))\u00b2 - 1
        //
        // caps at 20 iterations as a safety net against degenerate inputs.
        let mut xm = 1.0;
        let mut ym = 1.0;
        let mut zm = 1.0;
        for _ in 0..20 {
            xm = 1.0 / (1.0 + lambda * oor2.x);
            ym = 1.0 / (1.0 + lambda * oor2.y);
            zm = 1.0 / (1.0 + lambda * oor2.z);

            let xm2 = xm * xm;
            let ym2 = ym * ym;
            let zm2 = zm * zm;

            let f = x2 * xm2 + y2 * ym2 + z2 * zm2 - 1.0;
            if f.abs() < 1e-14 {
                break;
            }
            let df =
                -2.0 * (x2 * xm2 * xm * oor2.x + y2 * ym2 * ym * oor2.y + z2 * zm2 * zm * oor2.z);
            if df.abs() < f64::EPSILON {
                break;
            }
            lambda -= f / df;
        }

        Some(DVec3::new(ecef.x * xm, ecef.y * ym, ecef.z * zm))
    }

    /// Project an ECEF point onto the ellipsoid surface along the *geocentric*
    /// (not geodetic) normal - i.e., simple radial scaling.
    pub fn scale_to_geocentric_surface(&self, ecef: DVec3) -> Option<DVec3> {
        let mag2 = (ecef * self.one_over_radii).length_squared();
        if mag2 < 1e-30 {
            return None;
        }
        Some(ecef / mag2.sqrt())
    }

    /// First eccentricity squared: `e^2 = (a^2 − b^2) / a^2`.
    #[inline]
    pub fn first_eccentricity_squared(&self) -> f64 {
        (self.radii_squared.x - self.radii_squared.z) / self.radii_squared.x
    }

    /// Semi-major axis `a` (equatorial radius, metres).
    #[inline]
    pub fn semi_major_axis(&self) -> f64 {
        self.radii.x
    }

    /// Semi-minor axis `b` (polar radius, metres).
    #[inline]
    pub fn semi_minor_axis(&self) -> f64 {
        self.radii.z
    }

    /// Maximum radius of the ellipsoid (semi-major axis for oblate).
    #[inline]
    pub fn maximum_radius(&self) -> f64 {
        self.radii.x.max(self.radii.y).max(self.radii.z)
    }

    /// Minimum radius of the ellipsoid (semi-minor axis for oblate).
    #[inline]
    pub fn minimum_radius(&self) -> f64 {
        self.radii.x.min(self.radii.y).min(self.radii.z)
    }

    /// Transform an ECEF position to the scaled space of the ellipsoid
    /// (element-wise multiplication by `one_over_radii`).
    /// In scaled space, the ellipsoid becomes a unit sphere.
    #[inline]
    pub fn transform_position_to_scaled_space(&self, ecef: DVec3) -> DVec3 {
        ecef * self.one_over_radii
    }

    /// Transform a position from the ellipsoid's scaled space back to ECEF.
    /// Inverse of `transform_position_to_scaled_space`.
    #[inline]
    pub fn transform_position_from_scaled_space(&self, scaled: DVec3) -> DVec3 {
        scaled * self.radii
    }

    /// Compute the geodesic distance (metres) and azimuths between two geodetic
    /// positions using Vincenty's inverse formula.
    ///
    /// Returns `None` for (nearly) antipodal points where the solution is
    /// numerically undefined.
    ///
    /// Returns `(distance_m, initial_azimuth_rad, final_azimuth_rad)`.
    pub fn vincenty_inverse(
        &self,
        from: Cartographic,
        to: Cartographic,
    ) -> Option<(f64, f64, f64)> {
        let a = self.radii.x;
        let b = self.radii.z;
        let f = (a - b) / a;

        let lat1 = from.latitude;
        let lat2 = to.latitude;
        let lon1 = from.longitude;
        let lon2 = to.longitude;

        let u1 = ((1.0 - f) * lat1.tan()).atan();
        let u2 = ((1.0 - f) * lat2.tan()).atan();
        let l = (lon2 - lon1 + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;

        let sin_u1 = u1.sin();
        let cos_u1 = u1.cos();
        let sin_u2 = u2.sin();
        let cos_u2 = u2.cos();

        let mut lambda = l;
        let mut lambda_prev;
        let mut sin_sigma = 0.0_f64;
        let mut cos_sigma = 0.0_f64;
        let mut sigma = 0.0_f64;
        let mut sin_alpha;
        let mut cos2_alpha = 0.0_f64;
        let mut cos2_sigma_m = 0.0_f64;
        let mut converged = false;

        for _ in 0..100 {
            // Divergence guard
            if lambda.is_nan() || lambda.abs() > std::f64::consts::PI {
                return None;
            }

            let sin_lambda = lambda.sin();
            let cos_lambda = lambda.cos();

            sin_sigma = ((cos_u2 * sin_lambda) * (cos_u2 * sin_lambda)
                + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda)
                    * (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda))
                .sqrt();

            if sin_sigma == 0.0 {
                // coincident points
                return Some((0.0, 0.0, 0.0));
            }

            cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
            sigma = sin_sigma.atan2(cos_sigma);
            sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
            cos2_alpha = 1.0 - sin_alpha * sin_alpha;

            cos2_sigma_m = if cos2_alpha.abs() < 1e-15 {
                0.0 // equatorial line
            } else {
                cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
            };

            let c = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha));
            lambda_prev = lambda;
            lambda = l
                + (1.0 - c)
                    * f
                    * sin_alpha
                    * (sigma
                        + c * sin_sigma
                            * (cos2_sigma_m
                                + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));

            if (lambda - lambda_prev).abs() < 1e-12 {
                converged = true;
                break;
            }
        }

        if !converged {
            return None;
        }

        let u2_val = cos2_alpha * (a * a - b * b) / (b * b);
        let big_a = 1.0
            + u2_val / 16384.0 * (4096.0 + u2_val * (-768.0 + u2_val * (320.0 - 175.0 * u2_val)));
        let big_b = u2_val / 1024.0 * (256.0 + u2_val * (-128.0 + u2_val * (74.0 - 47.0 * u2_val)));
        let delta_sigma = big_b
            * sin_sigma
            * (cos2_sigma_m
                + big_b / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                        - big_b / 6.0
                            * cos2_sigma_m
                            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                            * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));

        let distance = b * big_a * (sigma - delta_sigma);

        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();
        let fwd_az = (cos_u2 * sin_lambda).atan2(cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda);
        let rev_az = (cos_u1 * sin_lambda).atan2(-sin_u1 * cos_u2 + cos_u1 * sin_u2 * cos_lambda);

        Some((distance, fwd_az, rev_az))
    }

    /// Compute the destination point given a start position, initial azimuth,
    /// and distance using Vincenty's direct formula.
    ///
    /// Returns `Some((destination, final_azimuth_rad))` on convergence, or
    /// `None` if the iteration fails to converge (e.g. near-antipodal inputs
    /// or NaN in intermediate values).
    pub fn vincenty_direct(
        &self,
        from: Cartographic,
        azimuth_rad: f64,
        distance_m: f64,
    ) -> Option<(Cartographic, f64)> {
        let a = self.radii.x;
        let b = self.radii.z;
        let f = (a - b) / a;

        let sin_az1 = azimuth_rad.sin();
        let cos_az1 = azimuth_rad.cos();

        let tan_u1 = (1.0 - f) * from.latitude.tan();
        let cos_u1 = 1.0 / (1.0 + tan_u1 * tan_u1).sqrt();
        let sin_u1 = tan_u1 * cos_u1;

        let sigma1 = tan_u1.atan2(cos_az1);
        let sin_alpha = cos_u1 * sin_az1;
        let cos2_alpha = 1.0 - sin_alpha * sin_alpha;

        let u2 = cos2_alpha * (a * a - b * b) / (b * b);
        let big_a = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)));
        let big_b = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)));

        let mut sigma = distance_m / (b * big_a);
        let mut sigma_prev;
        let mut converged = false;

        for _ in 0..200 {
            let cos2_sigma_m = (2.0 * sigma1 + sigma).cos();
            let delta_sigma = big_b
                * sigma.sin()
                * (cos2_sigma_m
                    + big_b / 4.0
                        * (sigma.cos() * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                            - big_b / 6.0
                                * cos2_sigma_m
                                * (-3.0 + 4.0 * sigma.sin() * sigma.sin())
                                * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));
            if delta_sigma.is_nan() {
                break;
            }
            sigma_prev = sigma;
            sigma = distance_m / (b * big_a) + delta_sigma;
            if (sigma - sigma_prev).abs() < 1e-12 {
                converged = true;
                break;
            }
        }

        if !converged {
            return None;
        }

        let cos2_sigma_m = (2.0 * sigma1 + sigma).cos();
        let x = sin_u1 * sigma.sin() - cos_u1 * sigma.cos() * cos_az1;
        let lat2 = (sin_u1 * sigma.cos() + cos_u1 * sigma.sin() * cos_az1)
            .atan2((1.0 - f) * (sin_alpha * sin_alpha + x * x).sqrt());
        let lambda_val =
            (sigma.sin() * sin_az1).atan2(cos_u1 * sigma.cos() - sin_u1 * sigma.sin() * cos_az1);
        let c = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha));
        let l = lambda_val
            - (1.0 - c)
                * f
                * sin_alpha
                * (sigma
                    + c * sigma.sin()
                        * (cos2_sigma_m
                            + c * sigma.cos() * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));
        let lon2 = (from.longitude + l + std::f64::consts::PI)
            .rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        let az2 = sin_alpha.atan2(-x);

        Some((Cartographic::new(lon2, lat2, from.height), az2))
    }

    /// Great-circle distance approximation using the Haversine formula
    /// (spherical Earth). Cheaper than Vincenty when moderate accuracy is
    /// acceptable.
    ///
    /// Uses the semi-major axis `a` as the sphere radius, matching CesiumJS
    /// (`EARTH_CIRCUMFERENCE / (2\pi) = a`).
    pub fn haversine_distance(&self, from: Cartographic, to: Cartographic) -> f64 {
        let r = self.radii.x;

        let dlat = to.latitude - from.latitude;
        let dlon = (to.longitude - from.longitude + std::f64::consts::PI)
            .rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        let h = (dlat / 2.0).sin().powi(2)
            + from.latitude.cos() * to.latitude.cos() * (dlon / 2.0).sin().powi(2);
        2.0 * r * h.sqrt().asin()
    }
}

/// A plane tangent to an [`Ellipsoid`] at a given surface point.
///
/// The origin is first projected onto the ellipsoid surface before the local
/// frame is computed; if the input origin is already on the surface the result
/// is exact.
///
/// # Example
/// ```
/// # use terra::{Ellipsoid, Cartographic, EllipsoidTangentPlane};
/// let ellipsoid = Ellipsoid::wgs84();
/// let origin = ellipsoid.cartographic_to_ecef(Cartographic::from_degrees(0.0, 0.0, 0.0));
/// let plane = EllipsoidTangentPlane::from_origin(origin, &ellipsoid).unwrap();
/// let pt_2d = plane.project_point_to_nearest_on_plane(origin);
/// assert!(pt_2d.x.abs() < 1e-6 && pt_2d.y.abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct EllipsoidTangentPlane {
    /// The origin expressed in ECEF - the surface projection of the input
    /// origin.
    origin: DVec3,
    /// East direction in ECEF.
    x_axis: DVec3,
    /// North direction in ECEF.
    y_axis: DVec3,
    /// Geodetic surface normal (= local Up = plane normal pointing outward).
    normal: DVec3,
    /// The underlying infinite plane in Hessian normal form.
    plane: Plane,
}

impl EllipsoidTangentPlane {
    /// Construct a tangent plane at the surface projection of `origin`.
    ///
    /// Returns `None` if `origin` is too close to the ellipsoid centre for the
    /// surface normal to be defined (same condition as
    /// [`Ellipsoid::scale_to_geodetic_surface`]).
    pub fn from_origin(origin: DVec3, ellipsoid: &Ellipsoid) -> Option<Self> {
        let surface_origin = ellipsoid.scale_to_geodetic_surface(origin)?;
        Some(Self::from_enu_matrix(east_north_up_to_ecef(
            surface_origin,
            ellipsoid,
        )))
    }

    /// Construct from a pre-computed East-North-Up -> ECEF transform matrix.
    ///
    /// The translation column is used as the origin; the first three columns
    /// supply the East, North, and Up axes.
    pub fn from_enu_matrix(enu_to_ecef: DMat4) -> Self {
        let origin = enu_to_ecef.col(3).truncate();
        let x_axis = enu_to_ecef.col(0).truncate().normalize();
        let y_axis = enu_to_ecef.col(1).truncate().normalize();
        let normal = enu_to_ecef.col(2).truncate().normalize();
        let plane = Plane::from_point_normal(origin, normal);
        Self {
            origin,
            x_axis,
            y_axis,
            normal,
            plane,
        }
    }

    /// The origin in ECEF (surface-projected).
    #[inline]
    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    /// East direction in ECEF.
    #[inline]
    pub fn x_axis(&self) -> DVec3 {
        self.x_axis
    }

    /// North direction in ECEF.
    #[inline]
    pub fn y_axis(&self) -> DVec3 {
        self.y_axis
    }

    /// Outward surface normal (= local Up).
    #[inline]
    pub fn normal(&self) -> DVec3 {
        self.normal
    }

    /// The underlying infinite plane.
    #[inline]
    pub fn plane(&self) -> &Plane {
        &self.plane
    }

    /// Project `point` (ECEF) along the plane normal onto this plane and return
    /// the resulting 2-D coordinates in the local (East, North) frame.
    ///
    /// Equivalent to Cesium's `EllipsoidTangentPlane.projectPointToNearestOnPlane`.
    #[inline]
    pub fn project_point_to_nearest_on_plane(&self, point: DVec3) -> DVec2 {
        let on_plane = self.plane.project_point(point);
        let diff = on_plane - self.origin;
        DVec2::new(diff.dot(self.x_axis), diff.dot(self.y_axis))
    }

    /// Project an iterator of ECEF points onto the plane, returning a `Vec` of
    /// 2-D coordinates.
    pub fn project_points_to_nearest_on_plane<I>(&self, points: I) -> Vec<DVec2>
    where
        I: IntoIterator<Item = DVec3>,
    {
        points
            .into_iter()
            .map(|p| self.project_point_to_nearest_on_plane(p))
            .collect()
    }

    /// Lift a 2-D plane coordinate back into ECEF space (on the tangent plane,
    /// not on the ellipsoid surface).
    ///
    /// The result lies on the infinite tangent plane, not the ellipsoid.
    /// Add height along `normal()` separately if needed.
    #[inline]
    pub fn unproject_point_onto_plane(&self, point2d: DVec2) -> DVec3 {
        self.origin + self.x_axis * point2d.x + self.y_axis * point2d.y
    }

    /// Convert a local (East, North) 2-D coordinate to an ECEF position on the
    /// plane, then project back onto the ellipsoid surface.
    ///
    /// Returns `None` if the elevation to the ellipsoid is undefined (point is
    /// very close to the centre of the Earth).
    pub fn unproject_point_onto_ellipsoid(
        &self,
        point2d: DVec2,
        ellipsoid: &Ellipsoid,
    ) -> Option<DVec3> {
        let on_plane = self.unproject_point_onto_plane(point2d);
        ellipsoid.scale_to_geodetic_surface(on_plane)
    }
}

/// A planar ellipsoid curve from a source to a destination point.
///
/// Create via [`SimplePlanarEllipsoidCurve::from_ecef`] or
/// [`SimplePlanarEllipsoidCurve::from_cartographic`].
///
/// Sample the curve with [`get_position`](Self::get_position).
///
/// # Algorithm
///
/// 1. Scale both ECEF endpoints to the geodetic surface (removes height).
/// 2. Record the rotation axis (`source x destination`) and total angle
///    between the two scaled positions.
/// 3. At percentage `t`, rotate the source direction by `t x total_angle`
///    around the rotation axis, then scale back to the ellipsoid surface and
///    add the linearly-interpolated height.
#[derive(Debug, Clone)]
pub struct SimplePlanarEllipsoidCurve {
    ellipsoid: Ellipsoid,
    /// Scaled (unit-surface) source direction.
    source_direction: DVec3,
    /// Rotation axis = source_scaled x destination_scaled.
    rotation_axis: DVec3,
    /// Total arc angle in radians.
    total_angle: f64,
    /// Height at the source.
    source_height: f64,
    /// Height at the destination.
    destination_height: f64,
}

impl SimplePlanarEllipsoidCurve {
    /// Build a curve between two ECEF positions.
    ///
    /// Returns `None` if either position cannot be projected onto the ellipsoid
    /// surface (e.g. the point is too close to the ellipsoid centre), or if the
    /// two surface projections are antipodal (rotation axis undefined).
    pub fn from_ecef(
        ellipsoid: &Ellipsoid,
        source_ecef: DVec3,
        destination_ecef: DVec3,
    ) -> Option<Self> {
        let source_surface = ellipsoid.scale_to_geodetic_surface(source_ecef)?;
        let destination_surface = ellipsoid.scale_to_geodetic_surface(destination_ecef)?;

        // Heights are distances from surface along the surface normal.
        let source_normal = ellipsoid.geodetic_surface_normal(source_surface);
        let dest_normal = ellipsoid.geodetic_surface_normal(destination_surface);
        let source_height = (source_ecef - source_surface).dot(source_normal);
        let destination_height = (destination_ecef - destination_surface).dot(dest_normal);

        // Scale both surface points to the unit sphere by dividing by the
        // ellipsoid radii.  For a surface point (x,y,z), x^2/a^2 + y^2/b^2 + z^2/c^2 = 1,
        // so (x/a, y/b, z/c) is already unit-length; we normalize anyway for
        // floating-point safety.
        let one_over_r = DVec3::new(
            1.0 / ellipsoid.radii.x,
            1.0 / ellipsoid.radii.y,
            1.0 / ellipsoid.radii.z,
        );
        let src_scaled = (source_surface * one_over_r).normalize();
        let dst_scaled = (destination_surface * one_over_r).normalize();

        let rotation_axis = src_scaled.cross(dst_scaled);
        let axis_len = rotation_axis.length();
        if axis_len < 1e-14 {
            // Points are coincident or antipodal.
            return None;
        }
        let rotation_axis = rotation_axis / axis_len;
        let total_angle = src_scaled.dot(dst_scaled).clamp(-1.0, 1.0).acos();

        Some(Self {
            ellipsoid: ellipsoid.clone(),
            source_direction: src_scaled,
            rotation_axis,
            total_angle,
            source_height,
            destination_height,
        })
    }

    /// Build a curve between two positions given as [`Cartographic`]
    /// (longitude, latitude, height).
    pub fn from_cartographic(
        ellipsoid: &Ellipsoid,
        source: Cartographic,
        destination: Cartographic,
    ) -> Option<Self> {
        let src_ecef = ellipsoid.cartographic_to_ecef(source);
        let dst_ecef = ellipsoid.cartographic_to_ecef(destination);
        Self::from_ecef(ellipsoid, src_ecef, dst_ecef)
    }

    /// Sample the curve at `percentage` (0 = source, 1 = destination).
    ///
    /// `percentage` is clamped to `[0, 1]`.
    ///
    /// `additional_height` is added on top of the linearly-interpolated height
    /// (useful for arc camera fly-to animations that want a mid-flight altitude
    /// boost).
    ///
    /// Returns the ECEF position of the sampled point.
    pub fn get_position(&self, percentage: f64, additional_height: f64) -> DVec3 {
        let t = percentage.clamp(0.0, 1.0);
        let angle = self.total_angle * t;

        // Rotate `source_direction` around `rotation_axis` by `angle`.
        let direction = rotate_vector_by_angle(self.source_direction, self.rotation_axis, angle);

        // Map the unit-sphere direction back to the ellipsoid surface.
        // The surface point satisfies: (x/a^2)^2 + (y/b^2)^2 + (z/c^2)^2 = 1
        // where direction is already normalised on the scaled sphere.
        // Scale back: multiply each component by the corresponding radius.
        let surface_ecef = direction * self.ellipsoid.radii;

        // Interpolate height and apply along the geodetic surface normal.
        let height = self.source_height
            + (self.destination_height - self.source_height) * t
            + additional_height;
        let normal = self.ellipsoid.geodetic_surface_normal(surface_ecef);
        surface_ecef + normal * height
    }

    /// Total arc angle in radians between source and destination.
    #[inline]
    pub fn total_angle(&self) -> f64 {
        self.total_angle
    }

    /// Height at the source endpoint.
    #[inline]
    pub fn source_height(&self) -> f64 {
        self.source_height
    }

    /// Height at the destination endpoint.
    #[inline]
    pub fn destination_height(&self) -> f64 {
        self.destination_height
    }
}

/// Rodrigues' rotation: rotate `v` around unit `axis` by `angle` radians.
#[inline]
fn rotate_vector_by_angle(v: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    let (sin, cos) = angle.sin_cos();
    // vxcos + (axis x v)xsin + axisx(axisxv)x(1 - cos)
    v * cos + axis.cross(v) * sin + axis * axis.dot(v) * (1.0 - cos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cartographic;
    use glam::DVec3;
    use std::f64::consts::PI;

    fn wgs84() -> Ellipsoid {
        Ellipsoid::wgs84()
    }

    fn carto(lon_deg: f64, lat_deg: f64, h: f64) -> Cartographic {
        Cartographic::from_degrees(lon_deg, lat_deg, h)
    }

    #[test]
    fn from_ecef_same_point_returns_none() {
        let e = wgs84();
        let pt = e.cartographic_to_ecef(carto(0.0, 0.0, 0.0));
        assert!(SimplePlanarEllipsoidCurve::from_ecef(&e, pt, pt).is_none());
    }

    #[test]
    fn start_matches_source() {
        let e = wgs84();
        let src = carto(0.0, 0.0, 1000.0);
        let dst = carto(10.0, 0.0, 2000.0);
        let curve = SimplePlanarEllipsoidCurve::from_cartographic(&e, src, dst).unwrap();
        let start = curve.get_position(0.0, 0.0);
        let expected = e.cartographic_to_ecef(src);
        let diff = (start - expected).length();
        assert!(diff < 1.0, "start mismatch: {diff}");
    }

    #[test]
    fn end_matches_destination() {
        let e = wgs84();
        let src = carto(0.0, 0.0, 0.0);
        let dst = carto(20.0, 10.0, 5000.0);
        let curve = SimplePlanarEllipsoidCurve::from_cartographic(&e, src, dst).unwrap();
        let end = curve.get_position(1.0, 0.0);
        let expected = e.cartographic_to_ecef(dst);
        let diff = (end - expected).length();
        assert!(diff < 1.0, "end mismatch: {diff}");
    }

    #[test]
    fn midpoint_height_is_average() {
        let e = wgs84();
        let src = carto(0.0, 0.0, 0.0);
        let dst = carto(10.0, 0.0, 1000.0);
        let curve = SimplePlanarEllipsoidCurve::from_cartographic(&e, src, dst).unwrap();
        let mid = curve.get_position(0.5, 0.0);
        let cartos = e.ecef_to_cartographic(mid).unwrap();
        let expected_height = 500.0;
        assert!(
            (cartos.height - expected_height).abs() < 0.5,
            "height={}",
            cartos.height
        );
    }

    #[test]
    fn additional_height_is_added() {
        let e = wgs84();
        let src = carto(0.0, 0.0, 0.0);
        let dst = carto(5.0, 0.0, 0.0);
        let curve = SimplePlanarEllipsoidCurve::from_cartographic(&e, src, dst).unwrap();
        let boost = 100_000.0_f64;
        let mid_base = curve.get_position(0.5, 0.0);
        let mid_boosted = curve.get_position(0.5, boost);
        let cartos_base = e.ecef_to_cartographic(mid_base).unwrap();
        let cartos_boosted = e.ecef_to_cartographic(mid_boosted).unwrap();
        let diff = cartos_boosted.height - cartos_base.height;
        assert!((diff - boost).abs() < 1.0, "boost diff={diff}");
    }

    #[test]
    fn percentage_clamped_below_zero() {
        let e = wgs84();
        let src = carto(0.0, 0.0, 100.0);
        let dst = carto(5.0, 0.0, 200.0);
        let curve = SimplePlanarEllipsoidCurve::from_cartographic(&e, src, dst).unwrap();
        let p0 = curve.get_position(0.0, 0.0);
        let p_neg = curve.get_position(-1.0, 0.0);
        assert!((p0 - p_neg).length() < 1.0);
    }

    #[test]
    fn percentage_clamped_above_one() {
        let e = wgs84();
        let src = carto(0.0, 0.0, 0.0);
        let dst = carto(5.0, 0.0, 0.0);
        let curve = SimplePlanarEllipsoidCurve::from_cartographic(&e, src, dst).unwrap();
        let p1 = curve.get_position(1.0, 0.0);
        let p2 = curve.get_position(2.0, 0.0);
        assert!((p1 - p2).length() < 1.0);
    }

    #[test]
    fn total_angle_positive() {
        let e = wgs84();
        let src = carto(0.0, 0.0, 0.0);
        let dst = carto(90.0, 0.0, 0.0);
        let curve = SimplePlanarEllipsoidCurve::from_cartographic(&e, src, dst).unwrap();
        // Quarter of the equator ~ PI/2 radians.
        assert!(
            (curve.total_angle() - PI / 2.0).abs() < 0.02,
            "angle={}",
            curve.total_angle()
        );
    }

    #[test]
    fn source_and_destination_heights_stored() {
        let e = wgs84();
        let src = carto(0.0, 0.0, 1000.0);
        let dst = carto(5.0, 0.0, 3000.0);
        let curve = SimplePlanarEllipsoidCurve::from_cartographic(&e, src, dst).unwrap();
        assert!((curve.source_height() - 1000.0).abs() < 1.0);
        assert!((curve.destination_height() - 3000.0).abs() < 1.0);
    }

    #[test]
    fn intermediate_points_lie_near_ellipsoid_at_zero_height() {
        let e = wgs84();
        let src = carto(0.0, 0.0, 0.0);
        let dst = carto(10.0, 0.0, 0.0);
        let curve = SimplePlanarEllipsoidCurve::from_cartographic(&e, src, dst).unwrap();
        for i in 1..=9 {
            let pt = curve.get_position(i as f64 / 10.0, 0.0);
            let c = e.ecef_to_cartographic(pt).unwrap();
            assert!(c.height.abs() < 0.5, "height at t={}: {}", i, c.height);
        }
    }

    fn wgs84_plane_at(lon_deg: f64, lat_deg: f64) -> EllipsoidTangentPlane {
        let e = Ellipsoid::wgs84();
        let origin = e.cartographic_to_ecef(Cartographic::from_degrees(lon_deg, lat_deg, 0.0));
        EllipsoidTangentPlane::from_origin(origin, &e).unwrap()
    }

    #[test]
    fn origin_projects_to_zero() {
        let plane = wgs84_plane_at(0.0, 0.0);
        let pt2d = plane.project_point_to_nearest_on_plane(plane.origin());
        assert!(pt2d.x.abs() < 1e-3, "x={}", pt2d.x);
        assert!(pt2d.y.abs() < 1e-3, "y={}", pt2d.y);
    }

    #[test]
    fn point_east_has_positive_x() {
        // A point one degree east of the origin should have positive x projection.
        let plane = wgs84_plane_at(0.0, 0.0);
        let e = Ellipsoid::wgs84();
        let east_pt = e.cartographic_to_ecef(Cartographic::from_degrees(1.0, 0.0, 0.0));
        let pt2d = plane.project_point_to_nearest_on_plane(east_pt);
        assert!(
            pt2d.x > 0.0,
            "east point should have positive x, got {}",
            pt2d.x
        );
        assert!(pt2d.y.abs() < 1000.0, "north component should be small");
    }

    #[test]
    fn point_north_has_positive_y() {
        let plane = wgs84_plane_at(0.0, 0.0);
        let e = Ellipsoid::wgs84();
        let north_pt = e.cartographic_to_ecef(Cartographic::from_degrees(0.0, 1.0, 0.0));
        let pt2d = plane.project_point_to_nearest_on_plane(north_pt);
        assert!(
            pt2d.y > 0.0,
            "north point should have positive y, got {}",
            pt2d.y
        );
        assert!(pt2d.x.abs() < 1000.0, "east component should be small");
    }

    #[test]
    fn unproject_roundtrip_on_plane() {
        let plane = wgs84_plane_at(10.0, 45.0);
        let local = DVec2::new(1000.0, -500.0);
        let ecef = plane.unproject_point_onto_plane(local);
        let back = plane.project_point_to_nearest_on_plane(ecef);
        assert!((back.x - local.x).abs() < 1e-6, "x roundtrip: {back}");
        assert!((back.y - local.y).abs() < 1e-6, "y roundtrip: {back}");
    }

    #[test]
    fn normal_is_unit_length() {
        let plane = wgs84_plane_at(45.0, 30.0);
        assert!((plane.normal().length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn axes_are_orthogonal() {
        let plane = wgs84_plane_at(45.0, 30.0);
        assert!(plane.x_axis().dot(plane.y_axis()).abs() < 1e-10);
        assert!(plane.x_axis().dot(plane.normal()).abs() < 1e-10);
        assert!(plane.y_axis().dot(plane.normal()).abs() < 1e-10);
    }

    #[test]
    fn from_enu_matrix_matches_from_origin() {
        let e = Ellipsoid::wgs84();
        let origin = e.cartographic_to_ecef(Cartographic::from_degrees(20.0, 50.0, 0.0));
        let enu = east_north_up_to_ecef(origin, &e);
        let p1 = EllipsoidTangentPlane::from_origin(origin, &e).unwrap();
        let p2 = EllipsoidTangentPlane::from_enu_matrix(enu);
        let diff = (p1.origin() - p2.origin()).length();
        assert!(diff < 1e-3, "origins differ by {diff}");
        let dot = p1.normal().dot(p2.normal());
        assert!((dot - 1.0).abs() < 1e-10, "normals differ: dot={dot}");
    }

    #[test]
    fn project_points_collects_all() {
        let plane = wgs84_plane_at(0.0, 0.0);
        let e = Ellipsoid::wgs84();
        let pts: Vec<DVec3> = (0..4)
            .map(|i| e.cartographic_to_ecef(Cartographic::from_degrees(i as f64, 0.0, 0.0)))
            .collect();
        let projected = plane.project_points_to_nearest_on_plane(pts.iter().copied());
        assert_eq!(projected.len(), 4);
    }

    #[test]
    fn unproject_onto_ellipsoid_returns_surface_point() {
        let e = Ellipsoid::wgs84();
        let plane = wgs84_plane_at(0.0, 0.0);
        let local = DVec2::new(0.0, 0.0);
        let surface = plane.unproject_point_onto_ellipsoid(local, &e).unwrap();
        let diff = (surface - plane.origin()).length();
        assert!(
            diff < 1.0,
            "surface point should be near origin, diff={diff}"
        );
    }

    /// Regression: `ecef_to_cartographic` previously returned the wrong sign
    /// for points inside the ellipsoid because it projected `h` onto the
    /// surface normal.
    #[test]
    fn ecef_to_cartographic_roundtrips_negative_heights() {
        let e = wgs84();
        for &(lon, lat, h) in &[
            (0.0_f64, 0.0, -1000.0),
            (45.0, 45.0, -500.0),
            (-120.0, 10.0, -10_000.0),
            (0.0, 89.0, -200.0),
            (0.0, 0.0, 100_000.0),
            (0.0, 0.0, 1_000_000.0),
        ] {
            let input = carto(lon, lat, h);
            let ecef = e.cartographic_to_ecef(input);
            let out = e.ecef_to_cartographic(ecef).expect("carto");
            assert!(
                (out.height - h).abs() < 1e-6,
                "height mismatch at (lon={lon}, lat={lat}, h={h}): got {}",
                out.height
            );
            assert!((out.longitude - input.longitude).abs() < 1e-9);
            assert!((out.latitude - input.latitude).abs() < 1e-9);
        }
    }

    // ---- vincenty_direct tests ----

    #[test]
    fn vincenty_direct_known_result() {
        // Paris -> ~111 km due North should arrive near lat 1°
        let e = wgs84();
        let from = carto(2.3522, 48.8566, 0.0);
        let (dest, _az2) = e
            .vincenty_direct(from, 0.0, 111_000.0)
            .expect("should converge");
        // Should end up ~1 degree of latitude further north
        let lat_diff = dest.latitude.to_degrees() - from.latitude.to_degrees();
        assert!(
            (lat_diff - 1.0).abs() < 0.01,
            "lat_diff={lat_diff:.4} expected ~1.0"
        );
    }

    #[test]
    fn vincenty_direct_longitude_wrapped() {
        // Starting near the antimeridian heading east; output lon must stay in [-PI, PI].
        let e = wgs84();
        let from = Cartographic::from_degrees(179.9, 0.0, 0.0);
        if let Some((dest, _)) = e.vincenty_direct(from, std::f64::consts::FRAC_PI_2, 50_000.0) {
            assert!(
                dest.longitude >= -std::f64::consts::PI && dest.longitude <= std::f64::consts::PI,
                "longitude out of range: {}",
                dest.longitude
            );
        }
    }

    // ---- vincenty_inverse tests ----

    #[test]
    fn vincenty_inverse_antipodal_returns_none() {
        let e = wgs84();
        // Equatorial antipodal: after longitude normalisation l = -PI, and
        // lambda crosses PI in the first iteration -> divergence guard fires.
        let from = carto(0.0, 0.0, 0.0);
        let to = carto(180.0, 0.0, 0.0);
        let result = e.vincenty_inverse(from, to);
        assert!(
            result.is_none(),
            "expected None for antipodal, got {result:?}"
        );
    }

    #[test]
    fn vincenty_inverse_known_distance() {
        let e = wgs84();
        // 1° of arc along the equator. Vincenty converges to exactly a·L for
        // the equatorial case (b/(1-f) = a).
        let (dist, _, _) = e
            .vincenty_inverse(carto(0.0, 0.0, 0.0), carto(1.0, 0.0, 0.0))
            .expect("should converge");
        let expected = Ellipsoid::WGS84_A * std::f64::consts::PI / 180.0;
        assert!(
            (dist - expected).abs() < 1.0,
            "distance={dist:.3} expected ~{expected:.3}"
        );
    }

    #[test]
    fn vincenty_inverse_dateline_longitude() {
        let e = wgs84();
        // Points on opposite sides of the antimeridian
        let a = carto(179.0, 0.0, 0.0);
        let b = carto(-179.0, 0.0, 0.0);
        let (dist, _, _) = e.vincenty_inverse(a, b).expect("should converge");
        // Should be ~222 km (2° of arc at the equator)
        assert!(dist < 250_000.0 && dist > 200_000.0, "distance={dist:.0}");
    }

    // ---- haversine_distance tests ----

    #[test]
    fn haversine_uses_semi_major_axis() {
        let e = wgs84();
        // A 90° arc on the equator: circumference/4
        let a = carto(0.0, 0.0, 0.0);
        let b = carto(90.0, 0.0, 0.0);
        let d = e.haversine_distance(a, b);
        let expected = std::f64::consts::FRAC_PI_2 * Ellipsoid::WGS84_A;
        assert!(
            (d - expected).abs() < 1.0,
            "d={d:.0} expected ~{expected:.0}"
        );
    }

    #[test]
    fn haversine_dateline_symmetric() {
        let e = wgs84();
        let a = carto(179.0, 0.0, 0.0);
        let b = carto(-179.0, 0.0, 0.0);
        let d = e.haversine_distance(a, b);
        // Same as going 2° the other way
        let c = carto(-179.0, 0.0, 0.0);
        let d2 = e.haversine_distance(b, a);
        // Both directions across the dateline should be equal
        assert!((d - d2).abs() < 1.0, "asymmetric: {d} vs {d2}");
        // And much shorter than going the long way around
        let long_way = e.haversine_distance(carto(0.0, 0.0, 0.0), carto(179.0, 0.0, 0.0));
        let _ = c;
        assert!(
            d < long_way,
            "dateline path should be shorter than long way"
        );
    }
}
