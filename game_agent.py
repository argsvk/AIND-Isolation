"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    filename='performance.log',
                    filemode='a+')
logger = logging.getLogger('game_agent')


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def negative_op(game, player):
    op_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return -op_moves


def conscious_chase(game, player):
    op_moves = set(game.get_legal_moves(game.get_opponent(player)))
    my_moves = set(game.get_legal_moves(player))
    same_moves = my_moves.intersection(op_moves)

    return len(my_moves) - len(same_moves)


def chase(game, player):
    op_moves = set(game.get_legal_moves(game.get_opponent(player)))
    my_moves = set(game.get_legal_moves(player))
    same_moves = my_moves.intersection(op_moves)

    return len(same_moves)


def considerate(game, player):
    op_moves = len(game.get_legal_moves(game.get_opponent(player)))
    my_moves = len(game.get_legal_moves(player))
    blank_spaces = len(game.get_blank_spaces())

    return 2 * my_moves - 0.5 * blank_spaces * op_moves


def op_aside(game, player):
    op_moves = len(game.get_legal_moves(game.get_opponent(player)))
    my_moves = len(game.get_legal_moves(player))

    return 2 * my_moves - op_moves


def center_routine(game, player):
    center = (game.width - 1) // 2, (game.height - 1) // 2
    op_loc = game.get_player_location(game.get_opponent(player))
    player_loc = game.get_player_location()

    i, j = center
    restricted = set((i + 1, j + 2), (i + 1, j - 2), (i + 2, j + 1),
                     (i + 2, j - 1), (i - 1, j + 2), (i - 1, j - 2),
                     (i - 2, j + 1), (i - 2, j - 1))

    if center == player_loc:
        return 100.0
    elif op_loc in restricted:
        mirror_move = player.mirror(game, player)
        if player.is_mirror(op_loc, player_loc):
            return 100.0
    elif op_loc == center:
        if player_loc in restricted:
            return 0.
        else:
            return 100.0

    return 0.


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    blanks = len(game.get_blank_spaces()) - 1

    ix = player.hash_function(game)
    score = player.hash_table[blanks][ix]
    hashed = score is True

    if not hashed:
        if game.move_count < 3:
            score = center_routine(game, player)
        if not score:
            # score = op_aside(game, player) + 1.0
            score = negative_op(game, player) + 1.0
            player.hash_table[blanks][ix] = score
        return score
    else:
        return player.hash_table[blanks][ix]


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.reflect = False
        self.PRIME = 1009
        self.hash_table = None

    def is_mirror(self, game, op_loc, player_loc):
        i, j = op_loc
        mirror_move = (game.height - 1 - i, game.width - 1 - j)

        return player_loc == mirror_move

    def hash_function(self, game):
        return game.hash() % self.PRIME

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        board_size = game.width * game.height

        if self.hash_table is None:
            self.hash_table = np.zeros((board_size, self.PRIME))
        if not legal_moves:
            return (-1, -1)
        elif len(legal_moves) == 1:
            return legal_moves[0]
        else:
            next_move = legal_moves[0]

        method = self.minimax if self.method == 'minimax' else self.alphabeta

        if self.iterative:
            depth = 1
            while True:
                try:
                    score, next_move = method(game, depth)
                    if time_left() <= self.TIMER_THRESHOLD:
                        return next_move
                    depth += 1

                except Timeout:
                    return next_move
        else:
            try:
                score, next_move = method(game, self.search_depth)
                return next_move

            except Timeout:
                return next_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if self.time_left() <= self.TIMER_THRESHOLD:
            raise Timeout()

        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = (-1, -1)
        legal_moves = game.get_legal_moves()

        for move in legal_moves:
            try:
                if depth == 1:
                    score = self.score(game.forecast_move(move), self)
                else:
                    score, _ = self.minimax(game.forecast_move(move),
                                            depth - 1, not maximizing_player)
                if maximizing_player:
                    if score > best_score:
                        best_score = score
                        best_move = move
                else:
                    if score < best_score:
                        best_score = score
                        best_move = move

            except Timeout:
                return best_score, best_move

        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        if maximizing_player:
            value = (float("-inf"), (-1, -1))
            for move in legal_moves:
                if depth == 1:
                    score = self.score(game.forecast_move(move), self), move
                else:
                    score = (self.alphabeta(game.forecast_move(move), depth - 1,
                             alpha, beta, not maximizing_player)[0], move)

                value = max([value, score], key=lambda x: x[0])
                alpha = max(alpha, value[0])

                if alpha >= beta:
                    break

        else:
            value = (float("inf"), (-1, -1))
            for move in legal_moves:
                if depth == 1:
                    score = self.score(game.forecast_move(move), self), move
                else:
                    score = (self.alphabeta(game.forecast_move(move), depth - 1,
                             alpha, beta, not maximizing_player)[0], move)
                value = min([value, score], key=lambda x: x[0])
                beta = min(beta, value[0])

                if beta <= alpha:
                    break

        return value
