![PyWiSim logo](diagrams/PyWiSim-logo260px.png)

[Back to README](../README.md)


# Code Style Guide

This document contains a basic style guide for writing Python code. PyWiSim follows this guide, which is essentially a simplified version of [Python PEP 8](https://peps.python.org/pep-0008/).


## Purpose

Guido van Rossum, the creator of Python, observed that “Code is read much more often than it’s written”. Readability of code is essential in any software project, but becomes even more important when this software is intended to be extended by other developers. Hence, PyWiSim is made readable by following the rules explained in this document.


## Code writing rules

### Naming

Choose sensible names which help figure out the purpose of an entity. Try to be concise but descriptive.

Type         | Naming convention     | Examples
------------ | ----------------------| --------
Function     | Lowercase only, separate with underscores | `push, run_qa_tests`
Variable     | Lowercase single letter, word or words, separate with underscores  | `x, counter, id_base`
Class        | Start each word with a capital, do not separate words (CamelCase) | `Scheduler, BaseStation`
Method       | Lowercase word or words, separate with underscores | `run, show_slices`
Constant     | Uppercase single letter, word or words, separate with underscore | `T, TIME, TIME_UNIT`
Module       | Lowercase word or words, separate with underscores | `simulator, qa_simulator`
Package      | Lowercase word or words, do not separate with underscores | `libsimnet, trafficgen`


### Blank lines

The following table indicates the number of blank lines which must precede different entities.

Type                                | Blank lines
----------------------------------- | -----------
Top-level functions, in modules     | 2
Classes                             | 3
Methods inside classes              | 2
Inside methods or functions, to separate steps | 1, use sparingly


### Maximum lenght line

PEP 8 suggests lines should be limited to 79 characters; this allows to open several files next to one another.

Within parenthesis `()`, brackets `[]` or braces `{}`, line continuation is assumed. Otherwise, separate lines with `\`:
```
def my_function(arg_1, arg_2,  # line break implied
        arg_3, arg_4):         # indent to improve readability
    return arg_1

from mypackage.my_module import ClassOne, \  # explicit line break
    function_1, ClassTwo, TIME_CONST         # indent to improve readability
```


### Indentation

Python requires indentation for code interpretation:
- use 4 consecutive spaces for indentation.
- use spaces, not tabs; maybe your editor can be configured to insert spaces with the TAB key.
- in line continuations, use indentation to improve readability.


### Block comments

Block comments are lines started by `#` which explain the purpose and functionality of a code block. Block comment rules:
- indent a block comment to the same level of the code it describes.
- start each line with `#` followed by a space.

An inline comment explains a single statement in a piece of code. Inline comment rules:
- write an inline comment in the same line as the statement it explains.
- separate the inline comment from the statement by two or more spaces.
- add a single space after the `#` which starts an inline comment.
- do no explain the obvious; good names can spare inline comments.


### Documentation strings

Documentation strings or _docstrings_ are strings enclosed in triple simple quotatuon marks (`'''`) or triple double quotation marks (`"""`) which appear on the first line of a function, method, class, or module. They explain the purpose of the entity they document, and can be accessed in the `.__doc__` attribute or through the `help()` function.

In PyWiSim, docstrings are written within simple triple quotes (`'''`) and follow the rules for [pydoctor](https://pydoctor.readthedocs.io/en/latest/index.html), a document generation application with which PyWiSim code documentation pages are created.


### Whitespace

Proper use of whitespaces improve readability of statements and expressions.

Surround binary operators with whitespaces:
- assignment operators: `x = 0`, `i += 1`.
- comparisons: `x == y`, `n >= 0`, `alfa in ls_names`, `a2 is b2`, `a2 is not b3`.
- booleans: `a and b`, `b or c`, `not d`.

Do not use whitespaces for default values in function arguments: `def fn(a1, a2, a3=0):`.

For several operators in a statement, only add whitespaces around operators with the lowest priority: `y = x**2 + p`, `z = (x+y) * (x-y)`, `if x>5 and x%2==0:`. This is also valid for the slice operator on lists: `a_list[3:7]`, `a_list[x+1 : x+2]`.

A comma, semicolon or colon must always be followed by a whitespace: `def fn(a1, a2, a3=0):`, `a, b = tuple_1`, `return n, N`. A comma, semicolon, or colon, must never be preceded by a whitespace.

Avoid a comma between the name of the function and the parenthesis that starts the arguments of the function, the brackets of a list, the braces of a dictionary or the parenthesis of a tuple; write like this: `func(a1, a2)`, `ls[1:]`, `dc = {'Peter':14}`, `a_tuple = (0, 0)`.

Avoid trailing spaces, i.e. whitespace at the end of a line; they are not visible and may produce differences on version control systems.


### Programming recommendations

There may be several ways to perform the same action. These suggestions tend to remove ambiguity and preserve consistency.

Do not compare boolean values to True or False with `==`; use the value itself like this: `if is_true:`.

The same with empty sequences or zero values, which are False in `if` statements: `if ls_items:`, `if message:`, `if var:` ; these will be True only if the list, string or variable are not empty or 0. 

To check if an argument is not None, use `if arg is not None:`. An empty list is not None, though it evaluates to False, as these statements show:
```
>>> arg = []

>>> if arg:
...     print(arg)
...

>>> if arg is not None:
...     print(arg)
...
[]
```

Use `.startswith()` and `.endswith()` to verify prefixes or suffixes on a string; this is clearer than slicing: `if message.startswith('cats'):`, `if message.endswith('dogs'):`, `if file_name.endswith('.jpg'):`.


## References

- Real Python. [How to Write Beautiful Python Code With PEP 8](https://realpython.com/python-pep8/). 
- Python. [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/).

[Back to README](../README.md)
