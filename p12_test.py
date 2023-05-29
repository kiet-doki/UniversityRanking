#!/usr/bin/python
import os, json, math
from collections import namedtuple
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

MAX_FILE_SIZE = 300 # units - KB
REL_TOL = 6e-04  # relative tolerance for floats
ABS_TOL = 15e-03  # absolute tolerance for floats

PASS = "PASS"

TEXT_FORMAT = "text"  # question type when expected answer is a str, int, float, or bool
TEXT_FORMAT_NAMEDTUPLE = "text namedtuple"  # question type when expected answer is a namedtuple
TEXT_FORMAT_UNORDERED_LIST = "text list_unordered"  # question type when the expected answer is a list where the order does *not* matter
TEXT_FORMAT_ORDERED_LIST = "text list_ordered"  # question type when the expected answer is a list where the order does matter
TEXT_FORMAT_SPECIAL_ORDERED_LIST = "text list_special_ordered"  # question type when the expected answer is a list where order does matter, but with possible ties. Elements are ordered according to values in special_ordered_json (with ties allowed)
TEXT_FORMAT_DICT = "text dict"  # question type when the expected answer is a dictionary
HTML_FORMAT = "html" # question type when the expected answer is a DataFrame
FILE_JSON_FORMAT = "file json" # question type when the expected answer is a JSON file

def return_expected_json():
    expected_json =    {"1": (TEXT_FORMAT, 103),
                        "2": (HTML_FORMAT, None),
                        "3": (HTML_FORMAT, None),
                        "4": (TEXT_FORMAT, 19),
                        "5": (TEXT_FORMAT, 87.26666666666667),
                        "6": (HTML_FORMAT, None),
                        "7": (TEXT_FORMAT, 'Ludwig Maximilian University of Munich'),
                        "8": (TEXT_FORMAT_UNORDERED_LIST, ['Harvard University',
                                                            'Massachusetts Institute of Technology',
                                                            'Stanford University',
                                                            'Columbia University',
                                                            'Princeton University',
                                                            'University of California, Berkeley',
                                                            'University of Pennsylvania',
                                                            'University of Chicago',
                                                            'California Institute of Technology',
                                                            'Yale University',
                                                            'Cornell University',
                                                            'Northwestern University',
                                                            'University of California, Los Angeles',
                                                            'University of Michigan, Ann Arbor',
                                                            'Johns Hopkins University',
                                                            'University of Washington - Seattle',
                                                            'University of Illinois at Urbana–Champaign',
                                                            'Duke University',
                                                            'University of Wisconsin–Madison',
                                                            'New York University',
                                                            'University of California San Diego',
                                                            'University of Texas at Austin',
                                                            'University of California, San Francisco',
                                                            'University of North Carolina at Chapel Hill',
                                                            'University of Minnesota - Twin Cities',
                                                            'University of Texas Southwestern Medical Center',
                                                            'Washington University in St. Louis',
                                                            'University of Southern California',
                                                            'Brown University',
                                                            'Vanderbilt University',
                                                            'Pennsylvania State University',
                                                            'Rutgers University–New Brunswick',
                                                            'Dartmouth College',
                                                            'University of California, Davis']),
                        "9": (TEXT_FORMAT, 'Fudan University'),
                        "10": (TEXT_FORMAT_ORDERED_LIST, ['Indian Institute of Science',
                                                            'Tata Institute of Fundamental Research',
                                                            'Indian Institute of Technology Bombay',
                                                            'University of Delhi',
                                                            'Indian Institute of Technology Madras']),
                        "11": (TEXT_FORMAT, 1856),
                        "institutions_df": (HTML_FORMAT, None),
                        "12": (TEXT_FORMAT_UNORDERED_LIST, ['SOAS University of London',
                                                            'Haverford College',
                                                            'Federal University of Mato Grosso do Sul',
                                                            'Tôn Đức Thắng University',
                                                            'International Institute for Management Development',
                                                            'Antioch College',
                                                            'USI - University of Italian Speaking Switzerland',
                                                            'École nationale supérieure de chimie de Montpellier',
                                                            'Academy of Scientific & Innovative Research']),
                        "13": (TEXT_FORMAT, 'USI - University of Italian Speaking Switzerland'),
                        "14": (TEXT_FORMAT, 451),
                        "15": (TEXT_FORMAT_UNORDERED_LIST, ["École nationale d'administration",
                                                            'INSEAD',
                                                            'HEC Paris',
                                                            'Institut Polytechnique de Paris',
                                                            'University of Tokyo',
                                                            'International Institute for Management Development',
                                                            'China Europe International Business School']),
                        "16": (TEXT_FORMAT_UNORDERED_LIST, ['Emory University',
                                                            'Aarhus University',
                                                            'Paris Diderot University',
                                                            'Paris-Sud University',
                                                            'École Polytechnique',
                                                            'University of California San Diego',
                                                            'École normale supérieure',
                                                            'Tufts University',
                                                            'University of Groningen',
                                                            'University of Texas MD Anderson Cancer Center',
                                                            'École Polytechnique Fédérale de Lausanne',
                                                            'University of Utah']),
                        "17": (TEXT_FORMAT_UNORDERED_LIST, ['USA', 'United Kingdom']),
                        "18": (TEXT_FORMAT_UNORDERED_LIST, ['World Rank',
                                                            'Institution',
                                                            'Country',
                                                            'National Rank',
                                                            'Quality of Education Rank',
                                                            'Alumni Employment Rank',
                                                            'Quality of Faculty Rank',
                                                            'Research Performance Rank',
                                                            'Score']),
                        "19": (TEXT_FORMAT_ORDERED_LIST, [{'Year': '2019-2020',
                                                              'World Rank': 1,
                                                              'Institution': 'Harvard University',
                                                              'Country': 'USA',
                                                              'National Rank': 1,
                                                              'Quality of Education Rank': 2,
                                                              'Alumni Employment Rank': 1,
                                                              'Quality of Faculty Rank': 1,
                                                              'Research Performance Rank': 1,
                                                              'Score': 100.0},
                                                             {'Year': '2019-2020',
                                                              'World Rank': 2,
                                                              'Institution': 'Massachusetts Institute of Technology',
                                                              'Country': 'USA',
                                                              'National Rank': 2,
                                                              'Quality of Education Rank': 1,
                                                              'Alumni Employment Rank': 10,
                                                              'Quality of Faculty Rank': 2,
                                                              'Research Performance Rank': 5,
                                                              'Score': 96.7},
                                                             {'Year': '2019-2020',
                                                              'World Rank': 3,
                                                              'Institution': 'Stanford University',
                                                              'Country': 'USA',
                                                              'National Rank': 3,
                                                              'Quality of Education Rank': 9,
                                                              'Alumni Employment Rank': 3,
                                                              'Quality of Faculty Rank': 3,
                                                              'Research Performance Rank': 2,
                                                              'Score': 95.2},
                                                             {'Year': '2019-2020',
                                                              'World Rank': 4,
                                                              'Institution': 'University of Cambridge',
                                                              'Country': 'United Kingdom',
                                                              'National Rank': 1,
                                                              'Quality of Education Rank': 4,
                                                              'Alumni Employment Rank': 19,
                                                              'Quality of Faculty Rank': 5,
                                                              'Research Performance Rank': 11,
                                                              'Score': 94.1},
                                                             {'Year': '2019-2020',
                                                              'World Rank': 5,
                                                              'Institution': 'University of Oxford',
                                                              'Country': 'United Kingdom',
                                                              'National Rank': 2,
                                                              'Quality of Education Rank': 10,
                                                              'Alumni Employment Rank': 24,
                                                              'Quality of Faculty Rank': 10,
                                                              'Research Performance Rank': 4,
                                                              'Score': 93.3}]),
                        "20": (FILE_JSON_FORMAT, "rankings.json")}


    return expected_json

def check_cell(qnum, actual):
    expected_json = return_expected_json()
    format, expected = expected_json[qnum[1:]]
    try:
        if format == TEXT_FORMAT:
            return simple_compare(expected, actual)
        elif format == TEXT_FORMAT_UNORDERED_LIST:
            return list_compare_unordered(expected, actual)
        elif format == TEXT_FORMAT_ORDERED_LIST:
            return list_compare_ordered(expected, actual)
        elif format == TEXT_FORMAT_DICT:
            return dict_compare(expected, actual)
        elif format == TEXT_FORMAT_NAMEDTUPLE:
            return namedtuple_compare(expected ,actual)
        elif format == HTML_FORMAT:
            return check_cell_html(qnum[1:], actual)
        elif format == FILE_JSON_FORMAT:
            return check_json(expected, actual)
        else:
            if expected != actual:
                return "expected %s but found %s " % (repr(expected), repr(actual))
    except:
        if expected != actual:
            return "expected %s" % (repr(expected))
    return PASS



def simple_compare(expected, actual, complete_msg=True):
    actual = getattr(actual, "tolist", lambda: actual)()
    msg = PASS
    if type(expected) == type:
        if expected != actual:
            if type(actual) == type:
                msg = "expected %s but found %s" % (expected.__name__, actual.__name__)
            else:
                msg = "expected %s but found %s" % (expected.__name__, repr(actual))
    elif type(expected) != type(actual) and not (type(expected) in [float, int] and type(actual) in [float, int]):
        msg = "expected to find type %s but found type %s" % (type(expected).__name__, type(actual).__name__)
    elif type(expected) == float:
        if not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            msg = "expected %s" % (repr(expected))
            if complete_msg:
                msg = msg + " but found %s" % (repr(actual))
    else:
        if expected != actual:
            msg = "expected %s" % (repr(expected))
            if complete_msg:
                msg = msg + " but found %s" % (repr(actual))
    return msg


def list_compare_ordered(expected, actual, obj="list"):
    msg = PASS
    if type(expected) != type(actual):
        msg = "expected to find type %s but found type %s" % (type(expected).__name__, type(actual).__name__)
        return msg
    for i in range(len(expected)):
        if i >= len(actual):
            msg = "expected missing %s in %s" % (repr(expected[i]), obj)
            break
        if type(expected[i]) in [int, float, bool, str]:
            val = simple_compare(expected[i], actual[i])
        elif type(expected[i]) in [list]:
            val = list_compare_ordered(expected[i], actual[i], "sub" + obj)
        elif type(expected[i]) in [dict]:
            val = dict_compare(expected[i], actual[i])
        elif type(expected[i]).__name__ in namedtuples:
            val = namedtuple_compare(expected[i], actual[i])
        if val != PASS:
            msg = "at index %d of the %s, " % (i, obj) + val
            break
    if len(actual) > len(expected) and msg == PASS:
        msg = "found unexpected %s in %s" % (repr(actual[len(expected)]), obj)
    if len(expected) != len(actual):
        msg = msg + " (found %d entries in %s, but expected %d)" % (len(actual), obj, len(expected))

    if len(expected) > 0 and type(expected[0]) in [int, float, bool, str]:
        if msg != PASS and list_compare_unordered(expected, actual, obj) == PASS:
            try:
                msg = msg + " (%s may not be ordered as required)" % (obj)
            except:
                pass
    return msg


def list_compare_helper(larger, smaller):
    msg = PASS
    j = 0
    for i in range(len(larger)):
        if i == len(smaller):
            msg = "expected %s" % (repr(larger[i]))
            break
        found = False
        while not found:
            if j == len(smaller):
                val = simple_compare(larger[i], smaller[j - 1], False)
                break
            val = simple_compare(larger[i], smaller[j], False)
            j += 1
            if val == PASS:
                found = True
                break
        if not found:
            msg = val
            break
    return msg


def list_compare_unordered(expected, actual, obj="list"):
    msg = PASS
    if type(expected) != type(actual):
        msg = "expected to find type %s but found type %s" % (type(expected).__name__, type(actual).__name__)
        return msg
    try:
        sort_expected = sorted(expected)
        sort_actual = sorted(actual)
    except:
        msg = "unexpected datatype found in %s; expected entries of type %s" % (obj, obj, type(expected[0]).__name__)
        return msg

    if len(actual) == 0 and len(expected) > 0:
        msg = "in the %s, missing" % (obj) + expected[0]
    elif len(actual) > 0 and len(expected) > 0:
        val = simple_compare(sort_expected[0], sort_actual[0])
        if val.startswith("expected to find type"):
            msg = "in the %s, " % (obj) + simple_compare(sort_expected[0], sort_actual[0])
        else:
            if len(expected) > len(actual):
                msg = "in the %s, missing " % (obj) + list_compare_helper(sort_expected, sort_actual)
            elif len(expected) < len(actual):
                msg = "in the %s, found un" % (obj) + list_compare_helper(sort_actual, sort_expected)
            if len(expected) != len(actual):
                msg = msg + " (found %d entries in %s, but expected %d)" % (len(actual), obj, len(expected))
                return msg
            else:
                val = list_compare_helper(sort_expected, sort_actual)
                if val != PASS:
                    msg = "in the %s, missing " % (obj) + val + ", but found un" + list_compare_helper(sort_actual,
                                                                                               sort_expected)
    return msg

def list_compare_special_init(expected, special_order):
    real_expected = []
    for i in range(len(expected)):
        if real_expected == [] or special_order[i-1] != special_order[i]:
            real_expected.append([])
        real_expected[-1].append(expected[i])
    return real_expected


def list_compare_special(expected, actual, special_order):
    expected = list_compare_special_init(expected, special_order)
    msg = PASS
    expected_list = []
    for expected_item in expected:
        expected_list.extend(expected_item)
    val = list_compare_unordered(expected_list, actual)
    if val != PASS:
        msg = val
    else:
        i = 0
        for expected_item in expected:
            j = len(expected_item)
            actual_item = actual[i: i + j]
            val = list_compare_unordered(expected_item, actual_item)
            if val != PASS:
                if j == 1:
                    msg = "at index %d " % (i) + val
                else:
                    msg = "between indices %d and %d " % (i, i + j - 1) + val
                msg = msg + " (list may not be ordered as required)"
                break
            i += j

    return msg


def dict_compare(expected, actual, obj="dict"):
    msg = PASS
    if type(expected) != type(actual):
        msg = "expected to find type %s but found type %s" % (type(expected).__name__, type(actual).__name__)
        return msg
    try:
        expected_keys = sorted(list(expected.keys()))
        actual_keys = sorted(list(actual.keys()))
    except:
        msg = "unexpected datatype found in keys of dict; expect a dict with keys of type %s" % (
            type(expected_keys[0]).__name__)
        return msg
    val = list_compare_unordered(expected_keys, actual_keys, "dict")
    if val != PASS:
        msg = "bad keys in %s: " % (obj) + val
    if msg == PASS:
        for key in expected:
            if expected[key] == None or type(expected[key]) in [int, float, bool, str]:
                val = simple_compare(expected[key], actual[key])
            elif type(expected[key]) in [list]:
                val = list_compare_ordered(expected[key], actual[key], "value")
            elif type(expected[key]) in [dict]:
                val = dict_compare(expected[key], actual[key], "sub" + obj)
            elif type(expected[key]).__name__ in namedtuples:
                val = namedtuple_compare(expected[key], actual[key])
            if val != PASS:
                msg = "incorrect val for key %s in %s: " % (repr(key), obj) + val
    return msg

def parse_df_html_table(html, question=None):
    soup = BeautifulSoup(html, 'html.parser')

    if question == None:
        tables = soup.find_all('table')
        assert(len(tables) == 1)
        table = tables[0]
    else:
        table = soup.find('table', {"data-question": str(question)})

    rows = []
    for tr in table.find_all('tr'):
        rows.append([])
        for cell in tr.find_all(['td', 'th']):
            rows[-1].append(cell.get_text())

    cells = {}
    for r in range(1, len(rows)):
        for c in range(1, len(rows[0])):
            rname = rows[r][0]
            cname = rows[0][c]
            cells[(rname,cname)] = rows[r][c]
    return cells

def check_cell_html(qnum, actual):
    try:
        actual_cells = parse_df_html_table(actual)
    except Exception as e:
        return "expected to find type DataFrame but found type %s instead" % type(actual).__name__
    try:
        with open('p12_expected.html', encoding='utf-8') as f:
            expected_cells = parse_df_html_table(f.read(), qnum)
    except Exception as e:
        return "ERROR! Could not find table in p12_expected.html. Please make sure you have downloaded p12_expected.html correctly."

    for location, expected in expected_cells.items():
        location_name = "column {} at index {}".format(location[1], location[0])
        actual = actual_cells.get(location, None)
        if actual == None:
            return "in location %s, expected to find %s" % (location_name, repr(expected))
        try:
            actual_ans = float(actual)
            expected_ans = float(expected)
            if math.isnan(actual_ans) and math.isnan(expected_ans):
                continue
        except Exception as e:
            actual_ans, expected_ans = actual, expected
        msg = simple_compare(expected_ans, actual_ans)
        if msg != PASS:
            return "in location %s, " % location_name + msg
    expected_cols = list(set(["column %s" %loc[1] for loc in expected_cells]))
    actual_cols = list(set(["column %s" %loc[1] for loc in actual_cells]))
    msg = list_compare_unordered(expected_cols, actual_cols, "DataFrame")
    if msg != PASS:
        return msg
    expected_rows = list(set(["row at index %s" %loc[0] for loc in expected_cells]))
    actual_rows = list(set(["row at index %s" %loc[0] for loc in actual_cells]))
    msg = list_compare_unordered(expected_rows, actual_rows, "DataFrame")
    if msg != PASS:
        return msg
    return PASS


def check_json(expected, actual):
    msg = PASS
    if expected not in os.listdir("."):
        return "file %s not found" % expected
    elif actual not in os.listdir("."):
        return "file %s not found" % actual
    try:
        e = open(expected, encoding='utf-8')
        expected_data = json.load(e)
        e.close()
    except json.JSONDecodeError:
        return "file %s is broken and cannot be parsed; please redownload the file" % expected
    try:
        a = open(actual, encoding='utf-8')
        actual_data = json.load(a)
        a.close()
    except json.JSONDecodeError:
        return "file %s is broken and cannot be parsed" % actual
    if type(expected_data) == list:
        msg = list_compare_ordered(expected_data, actual_data, 'file ' + actual)
    elif type(expected_data) == dict:
        msg = dict_compare(expected_data, actual_data)
    return msg

def check(qnum, actual):
    msg = check_cell(qnum, actual)
    if msg == PASS:
        return True
    print("<b style='color: red;'>ERROR:</b> " + msg)


def check_file_size(path):
    size = os.path.getsize(path)
    assert size < MAX_FILE_SIZE * 10**3, "Your file is too big to be processed by Gradescope; please delete unnecessary output cells so your file size is < %s KB" % MAX_FILE_SIZE
