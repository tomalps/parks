from xmlrpc.client import Boolean
import numpy as np
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt


class ParksPuzzle:
    def __init__(self, parks: np.ndarray, m: int, n: int) -> None:
        """Class for a Parks puzzle game.

        Args:
            parks (np.ndarray): nxn array of park locations going from 0 to n-1
            m (int): number of trees per park, row and column
            n (int): number of parks and size of grid
        """
        parks = np.asarray(parks)

        # check that the puzzle is valid
        assert parks.shape == (n, n)
        for i in range(m):
            assert i in parks
        for i in range(n):
            for j in range(n):
                assert parks[i, j] in range(n)

        # initialise class attributes
        self.parks = parks
        self.m = m
        self.n = n

        # nxn array of tree locations
        self.trees = np.zeros((n, n))

        # element k of P is a list of grid indices for park k
        self.P = [
            [(i, j) for i in range(n) for j in range(n) if parks[i, j] == k]
            for k in range(n)
        ]

    def solve(self) -> None:
        """Solve the puzzle."""
        m = self.m
        n = self.n
        P = self.P

        # Create the mip solver with the SCIP backend.
        solver = pywraplp.Solver.CreateSolver("SCIP")

        # Define the decision variables
        # x(i,j)=1 if there is a tree at grid indices (i,j)
        # otherwise x(i,j)=0
        x = {}
        for i in range(n):
            for j in range(n):
                x[i, j] = solver.BoolVar(f"x{i,j}")

        # Define the constraints
        # trees do not touch
        # i.e. each 2x2 subgrid contains no more than 1 tree
        for i in range(n - 1):
            for j in range(n - 1):
                solver.Add(
                    sum([x[i + d1, j + d2] for d1 in range(2) for d2 in range(2)]) <= 1
                )

        # m trees per row
        for i in range(n):
            solver.Add(sum([x[i, j] for j in range(n)]) == m)

        # m trees per column
        for j in range(n):
            solver.Add(sum([x[i, j] for i in range(n)]) == m)

        # m trees per park
        for k in range(n):
            solver.Add(sum([x[i, j] for (i, j) in P[k]]) == m)

        # Solve the problem
        status = solver.Solve()

        print("Number of variables =", solver.NumVariables())
        print("Number of constraints =", solver.NumConstraints())
        print()

        if status == pywraplp.Solver.OPTIMAL:
            print("Objective value =", solver.Objective().Value())
            for i in range(n):
                for j in range(n):
                    print(x[i, j].name(), " = ", x[i, j].solution_value())
            print()
            print("Problem solved in %f milliseconds" % solver.wall_time())
            print("Problem solved in %d iterations" % solver.iterations())
            print("Problem solved in %d branch-and-bound nodes" % solver.nodes())
        else:
            print("The problem does not have an optimal solution.")

        # Update trees grid
        self.trees = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if x[i, j].solution_value() > 0.5:
                    self.trees[i, j] = 1

    def draw(self, fname: str = None) -> None:
        """Draw parks and trees on grid.

        Args:
            fname (str, optional): Filename for saving plot. Defaults to None.
        """
        parks = self.parks
        n = self.n
        trees = self.trees

        cmap = plt.cm.Set3
        plt.figure()
        plt.imshow(parks, interpolation="nearest", cmap=cmap)

        xs, ys = np.ogrid[:n, :n]
        # the non-zero coordinates
        u = np.argwhere(trees)

        plt.scatter(
            ys[:, u[:, 1]].ravel(), xs[u[:, 0]].ravel(), marker="x", color="k", s=100
        )

        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        plt.show()


if __name__ == "__main__":
    # puzzle = ParksPuzzle(
    #     parks=[
    #         [0, 1, 1, 1, 1],
    #         [0, 0, 1, 1, 1],
    #         [0, 2, 3, 1, 1],
    #         [0, 3, 3, 3, 4],
    #         [3, 3, 3, 4, 4],
    #     ],
    #     m=1,
    #     n=5,
    # )
    # puzzle = ParksPuzzle(
    #     parks=[
    #         [0, 0, 1, 1, 2, 2, 1, 1],
    #         [0, 0, 0, 1, 1, 1, 1, 1],
    #         [0, 0, 0, 0, 1, 1, 3, 1],
    #         [0, 0, 0, 0, 1, 1, 3, 4],
    #         [0, 0, 0, 0, 0, 1, 3, 4],
    #         [0, 5, 0, 0, 0, 1, 3, 4],
    #         [0, 5, 6, 6, 6, 6, 6, 4],
    #         [0, 7, 7, 6, 6, 6, 6, 6],
    #     ],
    #     m=1,
    #     n=8,
    # )
    puzzle = ParksPuzzle(
        parks=[
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 2, 2, 2, 2],
            [0, 1, 1, 1, 1, 2, 3, 3, 3],
            [0, 0, 0, 4, 4, 4, 4, 4, 3],
            [5, 5, 0, 0, 0, 6, 6, 4, 3],
            [5, 0, 0, 7, 7, 6, 6, 4, 3],
            [5, 8, 8, 7, 7, 6, 6, 4, 3],
            [5, 8, 8, 7, 7, 8, 4, 4, 3],
            [5, 8, 8, 8, 8, 8, 4, 4, 3],
        ],
        m=2,
        n=9,
    )
    # puzzle = ParksPuzzle(
    #     parks=[
    #         [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
    #         [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
    #         [3, 1, 1, 1, 1, 1, 1, 4, 2, 2, 2, 2],
    #         [3, 3, 3, 3, 3, 1, 4, 4, 2, 4, 4, 2],
    #         [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5],
    #         [6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 5, 5],
    #         [6, 6, 6, 7, 3, 3, 8, 4, 8, 5, 5, 5],
    #         [6, 7, 7, 7, 3, 3, 8, 8, 8, 8, 5, 5],
    #         [9, 9, 9, 7, 7, 10, 10, 8, 8, 8, 5, 5],
    #         [9, 7, 7, 7, 7, 7, 10, 10, 8, 8, 5, 5],
    #         [7, 7, 7, 7, 11, 11, 11, 10, 8, 5, 5, 5],
    #         [7, 7, 7, 7, 11, 11, 11, 8, 8, 8, 5, 5],
    #     ],
    #     m=2,
    #     n=12,
    # )
    puzzle.solve()
    puzzle.draw(fname="nine_by_nine_solved.png")
