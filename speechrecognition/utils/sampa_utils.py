class SampaMapping:
    include = ['@', 'C', 'D', 'E', 'E:', 'E_?', 'I', 'I_?', 'N', 'O', 'OY', 'Q', 'R', 'S', 'U', 'Y', 'Z', '_2:', '_6',
               '_9', 'p:', 'p:', 'a', 'a:' 'aI', 'aU', 'au', 'b', 'c', 'd', 'e', 'e:', 'f', 'g', 'h', 'i', 'i:', 'j',
               'k', 'l', 'm', 'n', 'n:', 'o', 'o:', 'p', 'r', 's', 't', 'u', 'u:', 'v', 'w', 'x', 'y:', 'z']

    idx2sampa = {0: 'y:',
                 1: 't',
                 2: 'E:6',
                 3: 'b',
                 4: 'n',
                 5: 'p',
                 6: 'a:',
                 7: '2:',
                 8: 'v',
                 9: 'm',
                 10: 'U6',
                 11: 'k',
                 12: 'f',
                 13: 'u:6',
                 14: '2:6',
                 15: 'z',
                 16: 'Z',
                 17: 'E6',
                 18: 'I',
                 19: '~',
                 20: 'a',
                 21: 'x',
                 22: 'Q',
                 23: 'E:',
                 24: 'o:',
                 25: 'OU',
                 26: 'e:I',
                 27: 'd',
                 28: 'l',
                 29: 'aU',
                 30: 'q',
                 31: 'e:',
                 32: 'Oa',
                 33: 'a6',
                 34: 'u:',
                 35: 'w',
                 36: '@',
                 37: 'aI',
                 38: 'h',
                 39: '<p>',
                 40: '9',
                 41: 'S',
                 42: '?',
                 43: 'y:6',
                 44: 'i:',
                 45: 'OY',
                 46: 'N',
                 47: 'E',
                 48: 'Y6',
                 49: '6',
                 50: 'A',
                 51: 'j',
                 52: 'i:6',
                 53: 'g',
                 54: 'I6',
                 55: 'e:6',
                 56: 'r',
                 57: 'U',
                 58: 'i',
                 59: 'C',
                 60: '96',
                 61: 's',
                 62: 'O6',
                 63: 'O',
                 64: 'Y',
                 65: 'o:6',
                 66: 'a:6'}

    sampa2idx = {'2:': 7,
                 '2:6': 14,
                 '6': 49,
                 '9': 40,
                 '96': 60,
                 '<p>': 39,
                 '?': 42,
                 '@': 36,
                 'A': 50,
                 'C': 59,
                 'E': 47,
                 'E6': 17,
                 'E:': 23,
                 'E:6': 2,
                 'I': 18,
                 'I6': 54,
                 'N': 46,
                 'O': 63,
                 'O6': 62,
                 'OU': 25,
                 'OY': 45,
                 'Oa': 32,
                 'Q': 22,
                 'S': 41,
                 'U': 57,
                 'U6': 10,
                 'Y': 64,
                 'Y6': 48,
                 'Z': 16,
                 'a': 20,
                 'a6': 33,
                 'a:': 6,
                 'a:6': 66,
                 'aI': 37,
                 'aU': 29,
                 'b': 3,
                 'd': 27,
                 'e:': 31,
                 'e:6': 55,
                 'e:I': 26,
                 'f': 12,
                 'g': 53,
                 'h': 38,
                 'i': 58,
                 'i:': 44,
                 'i:6': 52,
                 'j': 51,
                 'k': 11,
                 'l': 28,
                 'm': 9,
                 'n': 4,
                 'o:': 24,
                 'o:6': 65,
                 'p': 5,
                 'q': 30,
                 'r': 56,
                 's': 61,
                 't': 1,
                 'u:': 34,
                 'u:6': 13,
                 'v': 8,
                 'w': 35,
                 'x': 21,
                 'y:': 0,
                 'y:6': 43,
                 'z': 15,
                 '~': 19}

    sampa2moa = {'2:': 'vowel',
                 '2:6': 'r',
                 '6': 'r',
                 '9': 'vowel',
                 '96': 'r',
                 '<p>': '<p>',
                 '?': 'fricative',
                 '@': 'vowel',
                 'C': 'fricative',
                 'A': 'vowel',
                 'E': 'vowel',
                 'E6': 'r',
                 'E:': 'vowel',
                 'E:6': 'r',
                 'I': 'vowel',
                 'I6': 'r',
                 'N': 'nasal',
                 'O': 'vowel',
                 'O6': 'r',
                 'OU': 'diphthong',
                 'OY': 'diphthong',
                 'Oa': 'diphthong',
                 'Q': 'vowel',
                 'S': 'vowel',
                 'U': 'vowel',
                 'U6': 'r',
                 'Y': 'vowel',
                 'Y6': 'r',
                 'Z': 'fricative',
                 'a': 'vowel',
                 'a6': 'r',
                 'a:': 'vowel',
                 'a:6': 'r',
                 'aI': 'diphthong',
                 'aU': 'diphthong',
                 'b': 'stop',
                 'd': 'stop',
                 'e:': 'vowel',
                 'e:6': 'r',
                 'e:I': 'diphthong',
                 'f': 'fricative',
                 'g': 'stop',
                 'h': 'fricative',
                 'i': 'vowel',
                 'i:': 'vowel',
                 'i:6': 'r',
                 'j': 'approximant',
                 'k': 'stop',
                 'l': 'approximant',
                 'm': 'nasal',
                 'n': 'nasal',
                 'o:': 'vowel',
                 'o:6': 'r',
                 'p': 'stop',
                 'q': 'stop',
                 'r': 'rhotic',
                 's': 'fricative',
                 't': 'stop',
                 'u:': 'vowel',
                 'u:6': 'diphthong',
                 'v': 'fricative',
                 'w': 'approximant',
                 'x': 'fricative',
                 'y:': 'vowel',
                 'y:6': 'r',
                 'z': 'fricative',
                 '~': 'nasal'}

    moa2idx = {'<p>': 8,
               'approximant': 3,
               'diphthong': 1,
               'fricative': 7,
               'nasal': 4,
               'r': 2,
               'rhotic': 5,
               'stop': 9,
               'vowel': 6}

    idx2moa = {1: 'diphthong',
               2: 'r',
               3: 'approximant',
               4: 'nasal',
               5: 'rhotic',
               6: 'vowel',
               7: 'fricative',
               8: '<p>',
               9: 'stop'}

    idx2poa = {1: 'vowel',
               2: 'diphthong',
               3: 'nasal',
               4: 'alveolar',
               5: 'labio-dental',
               6: '<p>',
               7: 'palatal',
               8: 'bilabial',
               9: 'post-alveolar',
               10: 'uvular',
               11: 'pharyngeal',
               12: 'r',
               13: 'velar',
               14: 'glottal'}

    poa2idx = {'<p>': 6,
               'alveolar': 4,
               'bilabial': 8,
               'diphthong': 2,
               'glottal': 14,
               'labio-dental': 5,
               'nasal': 3,
               'palatal': 7,
               'pharyngeal': 11,
               'post-alveolar': 9,
               'r': 12,
               'uvular': 10,
               'velar': 13,
               'vowel': 1}

    sampa2poa = {'2:': 'vowel',
                 '2:6': 'vowel',
                 '6': 'r',
                 '9': 'vowel',
                 '96': 'vowel',
                 '<p>': '<p>',
                 '?': 'pharyngeal',
                 '@': 'vowel',
                 'C': 'palatal',
                 'A': 'vowel',
                 'E': 'vowel',
                 'E6': 'r',
                 'E:': 'vowel',
                 'E:6': 'r',
                 'I': 'vowel',
                 'I6': 'r',
                 'N': 'velar',
                 'O': 'vowel',
                 'O6': 'r',
                 'OU': 'diphthong',
                 'OY': 'diphthong',
                 'Oa': 'diphthong',
                 'Q': 'vowel',
                 'S': 'vowel',
                 'U': 'vowel',
                 'U6': 'r',
                 'Y': 'vowel',
                 'Y6': 'r',
                 'Z': 'post-alveolar',
                 'a': 'vowel',
                 'a6': 'r',
                 'a:': 'vowel',
                 'a:6': 'r',
                 'aI': 'diphthong',
                 'aU': 'diphthong',
                 'b': 'bilabial',
                 'd': 'alveolar',
                 'e:': 'vowel',
                 'e:6': 'r',
                 'e:I': 'diphthong',
                 'f': 'labio-dental',
                 'g': 'velar',
                 'h': 'glottal',
                 'i': 'vowel',
                 'i:': 'vowel',
                 'i:6': 'r',
                 'j': 'palatal',
                 'k': 'velar',
                 'l': 'alveolar',
                 'm': 'bilabial',
                 'n': 'alveolar',
                 'o:': 'vowel',
                 'o:6': 'r',
                 'p': 'bilabial',
                 'q': 'uvular',
                 'r': 'alveolar',
                 's': 'alveolar',
                 't': 'alveolar',
                 'u:': 'vowel',
                 'u:6': 'diphthong',
                 'v': 'labio-dental',
                 'w': 'velar',
                 'x': 'velar',
                 'y:': 'vowel',
                 'y:6': 'r',
                 'z': 'alveolar',
                 '~': 'nasal'}

    sampa_correction = {
        '': '<p>',
        '!': '<p>',
        '"2:': '2:',
        '"2:"2:6': '2:6',
        '"2:"96': '96',
        '"9': '9',
        '"96': '96',
        '"E': 'E',
        '"E"e:': 'e:',
        '"E6': 'E6',
        '"E:': 'E:',
        '"E:"E:6': 'E:6',
        '"E:"e:': 'e:',
        '"E:"e:6': 'e:6',
        '"E:6': 'E:6',
        '"E:6"e:6': 'e:6',
        '"I': 'I',
        '"II': 'I',
        '"O': 'O',
        '"O6': 'O6',
        '"OY': 'OY',
        '"OYOY': 'OY',
        '"U': 'U',
        '"U6': 'U6',
        '"Y': 'Y',
        '"a': 'a',
        '"a6': 'a6',
        '"a6"a:6': 'a:6',
        '"a6a6': 'a6',
        '"a:': 'a:',
        '"a:"a': 'a',
        '"a:"a6': 'a6',
        '"a:"a:6': 'a:6',
        '"a:6': 'a:6',
        '"a:a:6': 'a:6',
        '"aI': 'aI',
        '"aU': 'aU',
        '"e:': 'e:',
        '"e:"e:6': 'e:6',
        '"e:6': 'e:6',
        '"e:6"e:': 'e:',
        '"i:': 'i:',
        '"i:"I': 'I',
        '"i:"i:6': 'i:6',
        '"i:6': 'i:6',
        '"o:': 'o:',
        '"o:"O': 'O',
        '"u:': 'u:',
        '"u:"U': 'U',
        '"y:': 'y:',
        '"y:"y:6': 'y:6',
        '"y:6': 'y:6',
        ',': '<p>',
        '.': '<p>',
        '/': '<p>',
        '2:': '2:',
        '2:2:6': '2:6',
        '2:6': '2:6',
        '2:96': '96',
        '6': '6',
        '6@': '6',
        '6E': '6',
        '6E6': 'E6',
        '6r': '6',
        '9': '9',
        '96': '96',
        '969': '9',
        ':k': 'k',
        ';': '<p>',
        '<p>': '<p>',
        '=/': '<p>',
        '=6': '6',
        '?': '?',
        '@': '@',
        '@E': 'E',
        '@I': 'I',
        '@e:': 'e:',
        'C': 'C',
        'CQ': 'C',
        'CS': 'C',
        'Cd': 'C',
        'Cf': 'C',
        'Cg': 'C',
        'Ch': 'C',
        'Cj': 'C',
        'Ck': 'C',
        'Cs': 'C',
        'E': 'E',
        'E6': 'E6',
        'E66': 'E6',
        'E6@': 'E6',
        'E6E': 'E6',
        'E6e:6': 'e:6',
        'E:': 'E:',
        'E:6': 'E:6',
        'E:6e:6': 'e:6',
        'E:E': 'E',
        'E:E:6': 'E:6',
        'E:e:': 'e:',
        'E:e:6': 'e:6',
        'E@': 'E',
        'EE': 'E',
        'EE6': 'E6',
        'EI': 'I',
        'Ee:': 'e:',
        'I': 'I',
        'I6': 'I6',
        'I6I': 'I6',
        'I6Y6': 'Y6',
        'I6i:6': 'i:6',
        'I@': 'I',
        'IE': 'E',
        'II': 'I',
        'IY': 'Y',
        'Ii:': 'i',
        'Il': 'I',
        'MA': 'A',
        'N': 'N',
        'Nm': 'm',
        'O': 'O',
        'O6': 'O6',
        'O6O': 'O6',
        'O6o:6': 'o:6',
        'O@': 'O',
        'OO': 'O',
        'OU': 'OU',
        'OY': 'OY',
        'Oa': 'Oa',
        'Q': 'Q',
        'S': 'S',
        'SZ': 'S',
        'Ss': 'S',
        'U': 'U',
        'U6': 'U6',
        'U6O6': 'U6',
        'U6U': 'U6',
        'U@': 'U',
        'Uu:': 'U',
        'Y': 'Y',
        'Y6': 'Y6',
        'Y6U6': 'U6',
        'Y6Y': 'Y6',
        'Y@': 'Y',
        'Z': 'Z',
        'ZS': 'Z',
        '_': '<p>',
        'a': 'a',
        'a6': 'a6',
        'a6a': 'a6',
        'a6a:': 'a6',
        'a6a:6': 'a:6',
        'a:': 'a:',
        'a:6': 'a:6',
        'a:6@': 'a:6',
        'a:6a': 'a:6',
        'a:6a:': 'a:6',
        'a:@': 'a:',
        'a:O': 'O',
        'a:a': 'a',
        'a:a6': 'a6',
        'a:a:': 'a:',
        'a:a:6': 'a:6',
        'a@': 'a',
        'aE': 'E',
        'aI': 'aI',
        'aI@': 'aI',
        'aIE': 'aI',
        'aIE:': 'aI',
        'aIa': 'aI',
        'aIa:': 'aI',
        'aU': 'aU',
        'aUE': 'aU',
        'aUO': 'aU',
        'aUa': 'aU',
        'aUaU': 'aU',
        'aUo:': 'aU',
        'aa': 'a',
        'aa:': 'a:',
        'a~': 'a',
        'b': 'b',
        'bm': 'b',
        'bp': 'b',
        'bv': 'b',
        'c:': '<p>',
        'd': 'd',
        'dN': 'd',
        'dQ': 'd',
        'dj': 'd',
        'dk': 'd',
        'dl': 'd',
        'dn': 'd',
        'dq': 'd',
        'ds': 'd',
        'dt': 'd',
        'dz': 'd',
        'e:': 'e:',
        'e:6': 'e:6',
        'e:66': 'e:6',
        'e:6@': 'e:6',
        'e:6E': 'e:6',
        'e:6E6': 'E6',
        'e:6e:': 'e:6',
        'e:@': 'e:',
        'e:E:': 'E:',
        'e:I': 'e:I',
        'e:e:': 'e:',
        'e:i:': 'i:',
        'f': 'f',
        'fs': 'f',
        'fv': 'f',
        'g': 'g',
        'gN': 'g',
        'gb': 'g',
        'gj': 'g',
        'gk': 'g',
        'gl': 'g',
        'gm': 'g',
        'gn': 'g',
        'gx': 'g',
        'h': 'h',
        'h:': 'h',
        'h:p:': 'p',
        'hC': 'h',
        'hx': 'h',
        'i:': 'i:',
        'i:6': 'i:6',
        'i:66': 'i:6',
        'i:6@': 'i:6',
        'i:6E': 'i:6',
        'i:6I': 'i:6',
        'i:6i:': 'i:6',
        'i:6y:': 'i:6',
        'i:@': 'i:',
        'i:I': 'I',
        'i:i:': 'i:',
        'i:i:6': 'i:6',
        'j': 'j',
        'jC': 'j',
        'jI': 'j',
        'ji:': 'j',
        'k': 'k',
        'kC': 'k',
        'kN': 'k',
        'kQ': 'k',
        'kf': 'k',
        'kg': 'k',
        'kh': 'k',
        'kpq': 'k',
        'kq': 'k',
        'kx': 'k',
        'l': 'l',
        'l:': 'l',
        'ld': 'l',
        'lj': 'l',
        'ln': 'l',
        'lr': 'l',
        'lt': 'l',
        'm': 'm',
        'mN': 'm',
        'mb': 'm',
        'mn': 'm',
        'n': 'n',
        'n:': 'n',
        'nN': 'n',
        'nd': 'n',
        'nl': 'n',
        'nm': 'n',
        'o:': 'o:',
        'o:"o:': 'o:',
        'o:6': 'o:6',
        'o:66': 'o:6',
        'o:6O': 'o:6',
        'o:6O6': 'o:6',
        'o:6U': 'o:6',
        'o:6o:': 'o:6',
        'o:@': 'o:',
        'o:O': 'O',
        'o:U': 'U',
        'o:o:6': 'o:6',
        'o:u:': 'u:',
        'p': 'p',
        'p:': 'p',
        'p:h:': 'p',
        'pb': 'p',
        'pf': 'p',
        'pq': 'p',
        'q': 'q',
        'r': 'r',
        'r:': 'r',
        's': 's',
        's:': 's',
        'sS': 's',
        'sh': 's',
        'sz': 's',
        't': 't',
        'tQ': 't',
        'tS': 't',
        'tb': 't',
        'td': 't',
        'tk': 't',
        'tn': 't',
        'tp': 't',
        'tq': 't',
        'ts': 't',
        'tz': 't',
        'u:': 'u:',
        'u:6': 'u:6',
        'u:6U6': 'u:6',
        'u:6u:': 'u:6',
        'u:@': 'u:',
        'u:U': 'U',
        'u:u:': 'u:',
        'u:u:6': 'u:6',
        'u:y:': 'y:',
        'v': 'v',
        'v:': 'v',
        'vb': 'v',
        'vf': 'v',
        'vh': 'v',
        'vm': 'v',
        'vz': 'v',
        'w:': 'w',
        'x': 'x',
        'xf': 'x',
        'xh': 'x',
        'xm': 'x',
        'xr': 'x',
        'y:': 'y:',
        'y:6': 'y:6',
        'y:6Y': 'y:6',
        'y:6Y6': 'Y6',
        'y:6y:': 'y:6',
        'y:6y:6': 'y:6',
        'y:@': 'y:',
        'y:Y': 'Y',
        'y:y:6': 'y:6',
        'z': 'z',
        'z:': 'z',
        'zs': 'z',
        'zt': 'z',
        '~': '~'
    }
