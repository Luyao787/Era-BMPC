import numpy as np
import numpy.linalg as LA

class PurePursuit:
    def __init__(self, wheel_base=3.0, K_dd=0.4,steering_max=np.pi/6):
        self.wheel_base = wheel_base
        self.K_dd = K_dd
        self.steering_max = steering_max
            
    def transform(self, pos, theta, waypoints):
        theta -=  np.pi/2
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return (LA.inv(rot) @ (waypoints - pos).T).T
    
    def compute_steering_angle(self, state_obsv, polylines, debug=False):
        look_ahead_distance = self.K_dd * state_obsv[3]        
            
        transformed_waypoints = self.transform(state_obsv[0:2], state_obsv[2], polylines)
        target_point = self.get_target_point(look_ahead_distance, transformed_waypoints)
    
        if target_point is None:
            return 0.   # steering angle
        alpha = np.arctan(-target_point[0] / target_point[1])
        # Change the steer output with the lateral controller.        
        steer = np.arctan((2 * self.wheel_base * np.sin(alpha)) / look_ahead_distance)
        steer = max(-self.steering_max, min(steer, self.steering_max))
        
        if debug:
            print(self.K_dd)
            print(state_obsv.speed)
            print(f"alpha: {alpha}")
            print(f"look_ahead_distance: {look_ahead_distance}")
            print("target point", target_point)
        
        return steer
    
    def get_target_point(self, lookahead, polyline):
        """ Determines the target point for the pure pursuit controller
        
        Parameters
        ----------
        lookahead : float
            The target point is on a circle of radius `lookahead`
            The circle's center is (0,0)
        poyline: array_like, shape (M,2)
            A list of 2d points that defines a polyline.
        
        Returns:
        --------
        target_point: numpy array, shape (,2)
            Point with positive x-coordinate where the circle of radius `lookahead`
            and the polyline intersect. 
            Return None if there is no such point.  
            If there are multiple such points, return the one that the polyline
            visits first.
        """
        intersections = []
        for j in range(len(polyline)-1):
            pt1 = polyline[j]
            pt2 = polyline[j+1]
            intersections += self.circle_line_segment_intersection((0,0), lookahead, pt1, pt2, full_line=False)
        # TODO:
        # for pt in intersections:
        #     if pt[1] > 0:
        #         return pt
        # return polyline[-1]
    
        filtered = [p for p in intersections if p[1]>0]
        if len(filtered)==0:
            return None
        return filtered[0]
    
    def circle_line_segment_intersection(self, circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
        # Function from https://stackoverflow.com/a/59582674/2609987
        """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

        :param circle_center: The (x, y) location of the circle center
        :param circle_radius: The radius of the circle
        :param pt1: The (x, y) location of the first point of the segment
        :param pt2: The (x, y) location of the second point of the segment
        :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
        :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
        :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

        Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
        """

        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)  
        dx, dy = (x2 - x1), (y2 - y1)
        dr = (dx ** 2 + dy ** 2)**.5
        big_d = x1 * y2 - x2 * y1
        discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

        if discriminant < 0:  # No intersection between circle and line
            return []
        else:  # There may be 0, 1, or 2 intersections with the segment            
            intersections = [
                (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
                cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
                for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
            if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
                fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
                intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
            if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
                return [intersections[0]]
            else:
                return intersections

