#!/usr/bin/env python3
# Copyright (c) 2008-11 Qtrac Ltd. All rights reserved.
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. It is provided for educational
# purposes and is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

import sys
import unicodedata


def print_unicode_table(words):
    print("decimal   hex   chr  {0:^40}".format("name"))
    print("-------  -----  ---  {0:-<40}".format(""))

    code = ord(" ")
    end = min(0xD800, sys.maxunicode)  # Stop at surrogate pairs
    while code < end:
        c = chr(code)
        name = unicodedata.name(c, "*** unknown ***")
        for i in range(0, len(words)):
            if words[i] not in name.lower():
                break
            elif i == len(words)-1:
                print("{0:7}  {0:5X}  {0:^3c}  {1}".format(
                    code, name.title()))
        code = code+1


words = []
if len(sys.argv) > 1:
    if sys.argv[1] in ("-h", "--help"):
        print("usage: {0} [string]".format(sys.argv[0]))
        word = 0
    else:
        for i in range(1, len(sys.argv)):
            words.append(sys.argv[i])
if len(words) != 0:
    print_unicode_table(words)
