from abc import ABCMeta, abstractmethod
from functools import lru_cache as cache
from decimal import Decimal, ROUND_HALF_UP
import numbers
from operator import itemgetter, gt, lt, add, sub
import itertools



DEFAULT_SNAP_TOLERANCE = 3
DEFAULT_JOIN_TOLERANCE = 3
DEFAULT_MIN_WORDS_VERTICAL = 3
DEFAULT_MIN_WORDS_HORIZONTAL = 1

class itemgetter:
    """
    Return a callable object that fetches the given item(s) from its operand.
    After f = itemgetter(2), the call f(r) returns r[2].
    After g = itemgetter(2, 5, 3), the call g(r) returns (r[2], r[5], r[3])
    """
    __slots__ = ('_items', '_call')

    def __init__(self, item, *items):
        if not items:
            self._items = (item,)
            def func(obj):
                return obj[item]
            self._call = func
        else:
            self._items = items = (item,) + items
            def func(obj):
                return tuple(obj[i] for i in items)
            self._call = func

    def __call__(self, obj):
        return self._call(obj)

    def __repr__(self):
        return '%s.%s(%s)' % (self.__class__.__module__,
                              self.__class__.__name__,
                              ', '.join(map(repr, self._items)))

    def __reduce__(self):
        return self.__class__, self._items



def edges_to_intersections(edges, x_tolerance=1, y_tolerance=1):
    """
    Given a list of edges, return the points at which they intersect
    within `tolerance` pixels.
    """
    intersections = {}
    v_edges, h_edges = [
        list(filter(lambda x: x["orientation"] == o, edges)) for o in ("v", "h")
    ]
    for v in sorted(v_edges, key=itemgetter("x0", "top")):
        for h in sorted(h_edges, key=itemgetter("top", "x0")):
            if (
                (v["top"] <= (h["top"] + y_tolerance))
                and (v["bottom"] >= (h["top"] - y_tolerance))
                and (v["x0"] >= (h["x0"] - x_tolerance))
                and (v["x0"] <= (h["x1"] + x_tolerance))
            ):
                vertex = (v["x0"], h["top"])
                if vertex not in intersections:
                    intersections[vertex] = {"v": [], "h": []}
                intersections[vertex]["v"].append(v)
                intersections[vertex]["h"].append(h)
    return intersections

def filter_edges(edges, orientation=None, edge_type=None, min_length=1):

    if orientation not in ("v", "h", None):
        raise ValueError("Orientation must be 'v' or 'h'")

    def test(e):
        dim = "height" if e["orientation"] == "v" else "width"
        et_correct = e["object_type"] == edge_type if edge_type is not None else True
        orient_correct = orientation is None or e["orientation"] == orientation
        return et_correct and orient_correct and (e[dim] >= min_length)

    edges = filter(test, edges)
    return list(edges)


class Integral():
    """Integral adds a conversion to int and the bit-string operations."""

    __slots__ = ()

    @abstractmethod
    def __int__(self):
        """int(self)"""
        raise NotImplementedError

    def __index__(self):
        """Called whenever an index is needed, such as in slicing"""
        return int(self)

    @abstractmethod
    def __pow__(self, exponent, modulus=None):
        """self ** exponent % modulus, but maybe faster.

        Accept the modulus argument if you want to support the
        3-argument version of pow(). Raise a TypeError if exponent < 0
        or any argument isn't Integral. Otherwise, just implement the
        2-argument version described in Complex.
        """
        raise NotImplementedError

    @abstractmethod
    def __lshift__(self, other):
        """self << other"""
        raise NotImplementedError

    @abstractmethod
    def __rlshift__(self, other):
        """other << self"""
        raise NotImplementedError

    @abstractmethod
    def __rshift__(self, other):
        """self >> other"""
        raise NotImplementedError

    @abstractmethod
    def __rrshift__(self, other):
        """other >> self"""
        raise NotImplementedError

    @abstractmethod
    def __and__(self, other):
        """self & other"""
        raise NotImplementedError

    @abstractmethod
    def __rand__(self, other):
        """other & self"""
        raise NotImplementedError

    @abstractmethod
    def __xor__(self, other):
        """self ^ other"""
        raise NotImplementedError

    @abstractmethod
    def __rxor__(self, other):
        """other ^ self"""
        raise NotImplementedError

    @abstractmethod
    def __or__(self, other):
        """self | other"""
        raise NotImplementedError

    @abstractmethod
    def __ror__(self, other):
        """other | self"""
        raise NotImplementedError

    @abstractmethod
    def __invert__(self):
        """~self"""
        raise NotImplementedError

    # Concrete implementations of Rational and Real abstract methods.
    def __float__(self):
        """float(self) == float(int(self))"""
        return float(int(self))

    @property
    def numerator(self):
        """Integers are their own numerators."""
        return +self

    @property
    def denominator(self):
        """Integers have a denominator of 1."""
        return 1


@cache(maxsize=int(10e4))
def _decimalize(v, q=None):
    # Convert int-like
    if isinstance(v, numbers.Integral):
        return Decimal(int(v))

    # Convert float-like
    elif isinstance(v, numbers.Real):
        if q is not None:
            return Decimal(repr(v)).quantize(Decimal(repr(q)), rounding=ROUND_HALF_UP)
        else:
            return Decimal(repr(v))
    else:
        raise ValueError(f"Cannot convert {v} to Decimal.")


def decimalize(v, q=None):
    # If already a decimal, just return itself
    if type(v) == Decimal:
        return v

    # If tuple/list passed, bulk-convert
    if isinstance(v, (tuple, list)):
        return type(v)(decimalize(x, q) for x in v)
    else:
        return _decimalize(v, q)

def curve_to_edges(curve):
    point_pairs = zip(curve["points"], curve["points"][1:])
    return [
        {
            "x0": min(p0[0], p1[0]),
            "x1": max(p0[0], p1[0]),
            "top": min(p0[1], p1[1]),
            "doctop": min(p0[1], p1[1]) + (curve["doctop"] - curve["top"]),
            "bottom": max(p0[1], p1[1]),
            "width": abs(p0[0] - p1[0]),
            "height": abs(p0[1] - p1[1]),
            "orientation": "v" if p0[0] == p1[0] else ("h" if p0[1] == p1[1] else None),
        }
        for p0, p1 in point_pairs
    ]


def rect_to_edges(rect):
    top, bottom, left, right = [dict(rect) for x in range(4)]
    top.update(
        {
            "object_type": "rect_edge",
            "height": decimalize(0),
            "y0": rect["y1"],
            "bottom": rect["top"],
            "orientation": "h",
        }
    )
    bottom.update(
        {
            "object_type": "rect_edge",
            "height": decimalize(0),
            "y1": rect["y0"],
            "top": rect["top"] + rect["height"],
            "doctop": rect["doctop"] + rect["height"],
            "orientation": "h",
        }
    )
    left.update(
        {
            "object_type": "rect_edge",
            "width": decimalize(0),
            "x1": rect["x0"],
            "orientation": "v",
        }
    )
    right.update(
        {
            "object_type": "rect_edge",
            "width": decimalize(0),
            "x0": rect["x1"],
            "orientation": "v",
        }
    )
    return [top, bottom, left, right]


def line_to_edge(line):
    edge = dict(line)
    edge["orientation"] = "h" if (line["top"] == line["bottom"]) else "v"
    return edge


def obj_to_edges(obj):
    return {
        "line": lambda x: [line_to_edge(x)],
        "rect": rect_to_edges,
        "rect_edge": rect_to_edges,
        "curve": curve_to_edges,
    }[obj["object_type"]](obj)

obj_to_bbox = itemgetter("x0", "top", "x1", "bottom")

def intersections_to_cells(intersections):
    """
    Given a list of points (`intersections`), return all rectangular "cells"
    that those points describe.

    `intersections` should be a dictionary with (x0, top) tuples as keys,
    and a list of edge objects as values. The edge objects should correspond
    to the edges that touch the intersection.
    """

    def edge_connects(p1, p2):
        def edges_to_set(edges):
            return set(map(obj_to_bbox, edges))

        if p1[0] == p2[0]:
            common = edges_to_set(intersections[p1]["v"]).intersection(
                edges_to_set(intersections[p2]["v"])
            )
            if len(common):
                return True

        if p1[1] == p2[1]:
            common = edges_to_set(intersections[p1]["h"]).intersection(
                edges_to_set(intersections[p2]["h"])
            )
            if len(common):
                return True
        return False

    points = list(sorted(intersections.keys()))
    n_points = len(points)

    def find_smallest_cell(points, i):
        if i == n_points - 1:
            return None
        pt = points[i]
        rest = points[i + 1 :]
        # Get all the points directly below and directly right
        below = [x for x in rest if x[0] == pt[0]]
        right = [x for x in rest if x[1] == pt[1]]
        for below_pt in below:
            if not edge_connects(pt, below_pt):
                continue

            for right_pt in right:
                if not edge_connects(pt, right_pt):
                    continue

                bottom_right = (right_pt[0], below_pt[1])

                if (
                    (bottom_right in intersections)
                    and edge_connects(bottom_right, right_pt)
                    and edge_connects(bottom_right, below_pt)
                ):

                    return (pt[0], pt[1], bottom_right[0], bottom_right[1])

    cell_gen = (find_smallest_cell(points, i) for i in range(len(points)))
    return list(filter(None, cell_gen))

def is_dataframe(collection):
    cls = collection.__class__
    name = ".".join([cls.__module__, cls.__name__])
    return name == "pandas.core.frame.DataFrame"


def to_list(collection):
    if is_dataframe(collection):
        return collection.to_dict("records")  # pragma: nocover
    else:
        return list(collection)


def cluster_list(xs, tolerance=0):
    tolerance = decimalize(tolerance)
    if tolerance == Decimal(0):
        return [[x] for x in sorted(xs)]
    if len(xs) < 2:
        return [[x] for x in sorted(xs)]
    groups = []
    xs = list(sorted(xs))
    current_group = [xs[0]]
    last = xs[0]
    for x in xs[1:]:
        if x <= (last + tolerance):
            current_group.append(x)
        else:
            groups.append(current_group)
            current_group = [x]
        last = x
    groups.append(current_group)
    return groups


def make_cluster_dict(values, tolerance):
    tolerance = decimalize(tolerance)
    clusters = cluster_list(set(values), tolerance)

    nested_tuples = [
        [(val, i) for val in value_cluster] for i, value_cluster in enumerate(clusters)
    ]

    cluster_dict = dict(itertools.chain(*nested_tuples))
    return cluster_dict



def cluster_objects(objs, attr, tolerance):
    if isinstance(attr, (str, int)):
        attr_getter = itemgetter(attr)
    else:
        attr_getter = attr
    objs = to_list(objs)
    values = map(attr_getter, objs)
    cluster_dict = make_cluster_dict(values, tolerance)

    get_0, get_1 = itemgetter(0), itemgetter(1)

    cluster_tuples = sorted(
        ((obj, cluster_dict.get(attr_getter(obj))) for obj in objs), key=get_1
    )

    grouped = itertools.groupby(cluster_tuples, key=get_1)

    clusters = [list(map(get_0, v)) for k, v in grouped]

    return clusters

def objects_to_bbox(objects):
    return (
        min(map(itemgetter("x0"), objects)),
        min(map(itemgetter("top"), objects)),
        max(map(itemgetter("x1"), objects)),
        max(map(itemgetter("bottom"), objects)),
    )


def get_bbox_overlap(a, b):
    a_left, a_top, a_right, a_bottom = decimalize(a)
    b_left, b_top, b_right, b_bottom = decimalize(b)
    o_left = max(a_left, b_left)
    o_right = min(a_right, b_right)
    o_bottom = min(a_bottom, b_bottom)
    o_top = max(a_top, b_top)
    o_width = o_right - o_left
    o_height = o_bottom - o_top
    if o_height >= 0 and o_width >= 0 and o_height + o_width > 0:
        return (o_left, o_top, o_right, o_bottom)
    else:
        return None

def bbox_to_rect(bbox):
    return {"x0": bbox[0], "top": bbox[1], "x1": bbox[2], "bottom": bbox[3]}

def objects_to_rect(objects):
    return {
        "x0": min(map(itemgetter("x0"), objects)),
        "x1": max(map(itemgetter("x1"), objects)),
        "top": min(map(itemgetter("top"), objects)),
        "bottom": max(map(itemgetter("bottom"), objects)),
    }


def words_to_edges_v(words, word_threshold=DEFAULT_MIN_WORDS_VERTICAL):
    """
    Find (imaginary) vertical lines that connect the left, right, or
    center of at least `word_threshold` words.
    """
    # Find words that share the same left, right, or centerpoints
    by_x0 = cluster_objects(words, "x0", 1)
    by_x1 = cluster_objects(words, "x1", 1)
    by_center = cluster_objects(words, lambda x: (x["x0"] + x["x1"]) / 2, 1)
    clusters = by_x0 + by_x1 + by_center

    # Find the points that align with the most words
    sorted_clusters = sorted(clusters, key=lambda x: -len(x))
    large_clusters = filter(lambda x: len(x) >= word_threshold, sorted_clusters)

    # For each of those points, find the bboxes fitting all matching words
    bboxes = list(map(objects_to_bbox, large_clusters))

    # Iterate through those bboxes, condensing overlapping bboxes
    condensed_bboxes = []
    for bbox in bboxes:
        overlap = False
        for c in condensed_bboxes:
            if get_bbox_overlap(bbox, c):
                overlap = True
                break
        if not overlap:
            condensed_bboxes.append(bbox)

    if len(condensed_bboxes) == 0:
        return []

    condensed_rects = map(bbox_to_rect, condensed_bboxes)
    sorted_rects = list(sorted(condensed_rects, key=itemgetter("x0")))

    max_x1 = max(map(itemgetter("x1"), sorted_rects))
    min_top = min(map(itemgetter("top"), sorted_rects))
    max_bottom = max(map(itemgetter("bottom"), sorted_rects))

    # Describe all the left-hand edges of each text cluster
    edges = [
        {
            "x0": b["x0"],
            "x1": b["x0"],
            "top": min_top,
            "bottom": max_bottom,
            "height": max_bottom - min_top,
            "orientation": "v",
        }
        for b in sorted_rects
    ] + [
        {
            "x0": max_x1,
            "x1": max_x1,
            "top": min_top,
            "bottom": max_bottom,
            "height": max_bottom - min_top,
            "orientation": "v",
        }
    ]

    return edges

def words_to_edges_h(words, word_threshold=DEFAULT_MIN_WORDS_HORIZONTAL):
    """
    Find (imaginary) horizontal lines that connect the tops
    of at least `word_threshold` words.
    """
    by_top = cluster_objects(words, "top", 1)
    large_clusters = filter(lambda x: len(x) >= word_threshold, by_top)
    rects = list(map(objects_to_rect, large_clusters))
    if len(rects) == 0:
        return []
    min_x0 = min(map(itemgetter("x0"), rects))
    max_x1 = max(map(itemgetter("x1"), rects))
    max_bottom = max(map(itemgetter("bottom"), rects))
    edges = [
        {
            "x0": min_x0,
            "x1": max_x1,
            "top": r["top"],
            "bottom": r["top"],
            "width": max_x1 - min_x0,
            "orientation": "h",
        }
        for r in rects
    ] + [
        {
            "x0": min_x0,
            "x1": max_x1,
            "top": max_bottom,
            "bottom": max_bottom,
            "width": max_x1 - min_x0,
            "orientation": "h",
        }
    ]

    return edges


def move_object(obj, axis, value):
    assert axis in ("h", "v")
    if axis == "h":
        new_items = (
            ("x0", obj["x0"] + value),
            ("x1", obj["x1"] + value),
        )
    if axis == "v":
        new_items = [
            ("top", obj["top"] + value),
            ("bottom", obj["bottom"] + value),
        ]
        if "doctop" in obj:
            new_items += [("doctop", obj["doctop"] + value)]
        if "y0" in obj:
            new_items += [
                ("y0", obj["y0"] - value),
                ("y1", obj["y1"] - value),
            ]
    return obj.__class__(tuple(obj.items()) + tuple(new_items))


def snap_objects(objs, attr, tolerance):
    axis = {"x0": "h", "x1": "h", "top": "v", "bottom": "v"}[attr]
    clusters = cluster_objects(objs, attr, tolerance)
    avgs = [sum(map(itemgetter(attr), objs)) / len(objs) for objs in clusters]
    snapped_clusters = [
        [move_object(obj, axis, avg - obj[attr]) for obj in cluster]
        for cluster, avg in zip(clusters, avgs)
    ]
    return list(itertools.chain(*snapped_clusters))


def snap_edges(edges, tolerance=DEFAULT_SNAP_TOLERANCE):
    """
    Given a list of edges, snap any within `tolerance` pixels of one another
    to their positional average.
    """
    v, h = [list(filter(lambda x: x["orientation"] == o, edges)) for o in ("v", "h")]

    snap = snap_objects
    snapped = snap(v, "x0", tolerance) + snap(h, "top", tolerance)
    return snapped





def merge_edges(edges, snap_tolerance, join_tolerance):
    """
    Using the `snap_edges` and `join_edge_group` methods above,
    merge a list of edges into a more "seamless" list.
    """

    def get_group(edge):
        if edge["orientation"] == "h":
            return ("h", edge["top"])
        else:
            return ("v", edge["x0"])

    if snap_tolerance > 0:
        edges = snap_edges(edges, snap_tolerance)

    if join_tolerance > 0:
        _sorted = sorted(edges, key=get_group)
        edge_groups = itertools.groupby(_sorted, key=get_group)
        edge_gen = (
            join_edge_group(items, k[0], join_tolerance) for k, items in edge_groups
        )
        edges = list(itertools.chain(*edge_gen))
    return edges


def resize_object(obj, key, value):
    assert key in ("x0", "x1", "top", "bottom")
    old_value = obj[key]
    diff = value - old_value
    new_items = [
        (key, value),
    ]
    if key == "x0":
        assert value <= obj["x1"]
        new_items.append(("width", obj["x1"] - value))
    elif key == "x1":
        assert value >= obj["x0"]
        new_items.append(("width", value - obj["x0"]))
    elif key == "top":
        assert value <= obj["bottom"]
        new_items.append(("doctop", obj["doctop"] + diff))
        new_items.append(("height", obj["height"] - diff))
        if "y1" in obj:
            new_items.append(("y1", obj["y1"] - diff))
    elif key == "bottom":
        assert value >= obj["top"]
        new_items.append(("height", obj["height"] + diff))
        if "y0" in obj:
            new_items.append(("y0", obj["y0"] - diff))
    return obj.__class__(tuple(obj.items()) + tuple(new_items))


def join_edge_group(edges, orientation, tolerance=DEFAULT_JOIN_TOLERANCE):
    """
    Given a list of edges along the same infinite line, join those that
    are within `tolerance` pixels of one another.
    """
    if orientation == "h":
        min_prop, max_prop = "x0", "x1"
    elif orientation == "v":
        min_prop, max_prop = "top", "bottom"
    else:
        raise ValueError("Orientation must be 'v' or 'h'")

    sorted_edges = list(sorted(edges, key=itemgetter(min_prop)))
    joined = [sorted_edges[0]]
    for e in sorted_edges[1:]:
        last = joined[-1]
        if e[min_prop] <= (last[max_prop] + tolerance):
            if e[max_prop] > last[max_prop]:
                # Extend current edge to new extremity
                joined[-1] = resize_object(last, max_prop, e[max_prop])
        else:
            # Edge is separate from previous edges
            joined.append(e)

    return joined



TABLE_STRATEGIES = ["lines", "lines_strict", "text", "explicit"]
DEFAULT_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "explicit_vertical_lines": [],
    "explicit_horizontal_lines": [],
    "snap_tolerance": DEFAULT_SNAP_TOLERANCE,
    "join_tolerance": DEFAULT_JOIN_TOLERANCE,
    "edge_min_length": 3,
    "min_words_vertical": DEFAULT_MIN_WORDS_VERTICAL,
    "min_words_horizontal": DEFAULT_MIN_WORDS_HORIZONTAL,
    "keep_blank_chars": False,
    "text_tolerance": 3,
    "text_x_tolerance": None,
    "text_y_tolerance": None,
    "intersection_tolerance": 3,
    "intersection_x_tolerance": None,
    "intersection_y_tolerance": None,
}



class TableFinder(object):
    """
    Given a PDF page, find plausible table structures.

    Largely borrowed from Anssi Nurminen's master's thesis:
    http://dspace.cc.tut.fi/dpub/bitstream/handle/123456789/21520/Nurminen.pdf?sequence=3

    ... and inspired by Tabula:
    https://github.com/tabulapdf/tabula-extractor/issues/16
    """

    def __init__(self, page, settings={}):
        for k in settings.keys():
            if k not in DEFAULT_TABLE_SETTINGS:
                raise ValueError(f"Unrecognized table setting: '{k}'")
        self.page = page
        self.settings = dict(DEFAULT_TABLE_SETTINGS)
        self.settings.update(settings)
        for var, fallback in [
            ("text_x_tolerance", "text_tolerance"),
            ("text_y_tolerance", "text_tolerance"),
            ("intersection_x_tolerance", "intersection_tolerance"),
            ("intersection_y_tolerance", "intersection_tolerance"),
        ]:
            if self.settings[var] is None:
                self.settings.update({var: self.settings[fallback]})
        self.edges = self.get_edges()
        self.intersections = edges_to_intersections(
            self.edges,
            self.settings["intersection_x_tolerance"],
            self.settings["intersection_y_tolerance"],
        )
        self.cells = intersections_to_cells(self.intersections)

    def get_edges(self):
        settings = self.settings
        for name in ["vertical", "horizontal"]:
            strategy = settings[name + "_strategy"]
            if strategy not in TABLE_STRATEGIES:
                raise ValueError(
                    f'{name}_strategy must be one of {{{",".join(TABLE_STRATEGIES)}}}'
                )
            if strategy == "explicit":
                if len(settings["explicit_" + name + "_lines"]) < 2:
                    raise ValueError(
                        f"If {strategy}_strategy == 'explicit', explicit_{name}_lines "
                        f"must be specified as a list/tuple of two or more "
                        f"floats/ints."
                    )

        v_strat = settings["vertical_strategy"]
        h_strat = settings["horizontal_strategy"]

        if v_strat == "text" or h_strat == "text":
            words = self.page.extract_words(
                x_tolerance=settings["text_x_tolerance"],
                y_tolerance=settings["text_y_tolerance"],
                keep_blank_chars=settings["keep_blank_chars"],
            )

        v_explicit = []
        for desc in settings["explicit_vertical_lines"]:
            if isinstance(desc, dict):
                for e in obj_to_edges(desc):
                    if e["orientation"] == "v":
                        v_explicit.append(e)
            else:
                v_explicit.append(
                    {
                        "x0": desc,
                        "x1": desc,
                        "top": self.page.bbox[1],
                        "bottom": self.page.bbox[3],
                        "height": self.page.bbox[3] - self.page.bbox[1],
                        "orientation": "v",
                    }
                )

        if v_strat == "lines":
            v_base = filter_edges(self.page.edges, "v")
        elif v_strat == "lines_strict":
            v_base = filter_edges(self.page.edges, "v", edge_type="line")
        elif v_strat == "text":
            v_base = words_to_edges_v(
                words, word_threshold=settings["min_words_vertical"]
            )
        elif v_strat == "explicit":
            v_base = []

        v = v_base + v_explicit

        h_explicit = []
        for desc in settings["explicit_horizontal_lines"]:
            if isinstance(desc, dict):
                for e in obj_to_edges(desc):
                    if e["orientation"] == "h":
                        h_explicit.append(e)
            else:
                h_explicit.append(
                    {
                        "x0": self.page.bbox[0],
                        "x1": self.page.bbox[2],
                        "width": self.page.bbox[2] - self.page.bbox[0],
                        "top": desc,
                        "bottom": desc,
                        "orientation": "h",
                    }
                )

        if h_strat == "lines":
            h_base = filter_edges(self.page.edges, "h")
        elif h_strat == "lines_strict":
            h_base = filter_edges(self.page.edges, "h", edge_type="line")
        elif h_strat == "text":
            h_base = words_to_edges_h(
                words, word_threshold=settings["min_words_horizontal"]
            )
        elif h_strat == "explicit":
            h_base = []

        h = h_base + h_explicit

        edges = list(v) + list(h)

        if settings["snap_tolerance"] > 0 or settings["join_tolerance"] > 0:
            edges = merge_edges(
                edges,
                snap_tolerance=settings["snap_tolerance"],
                join_tolerance=settings["join_tolerance"],
            )
        return filter_edges(edges, min_length=settings["edge_min_length"])


def find_table_coord(page, table_settings={}):
    return TableFinder(page, table_settings).cells