export const sortMap = (
};

export const NUM_MAP: Map<string, number> = sortMap(
  new Map([
    ['M', 1000],
    ['CM', 900],
    ['D', 500],
    ['CD', 400],
    ['C', 100],
    ['XC', 90],
    ['L', 50],
    ['XL', 40],
    ['X', 10],
    ['IX', 9],
    ['V', 5],
    ['IV', 4],
    ['I', 1],
  ]),
  'dsc'
);

export const ROMAN_NUMERALS: string[] = Array.from(NUM_MAP.keys()).filter(
  romanNumeral => romanNumeral.length === 1
);

/**
 * Converts given arabic number to roman.
 *
 * @param arabicNum - arabic number that needs to be converted to roman number
 * @returns roman number for given arabic number
 * @example
 * romanize(671);
 * // returns 'DCLXXI'
 */
export const romanize = (arabicNum: number | string): string => {
  let romanNum = '';

  if (typeof arabicNum === 'string') {
    arabicNum = Number(arabicNum);
  }

  if (!Number.isInteger(arabicNum) || arabicNum <= 0) {
    throw new Error('Number must be a positive integer.');
  } else if (arabicNum > 3999) {
    throw new Error(
      'The largest number that can be represented using roman numerals is 3999 (MMMCMXCIX).'
    );
  }

  for (const [numMapRoman, numMapArabic] of NUM_MAP) {
    if (arabicNum === 0) break;

    while (arabicNum >= numMapArabic) {
      romanNum += numMapRoman;
      arabicNum -= numMapArabic;
    }
  }

  return romanNum;
};

/**
 * Converts given roman number to arabic.
 *
 * @param romanNum - roman number that needs to be converted to arabic
 * @returns arabic number for given roman number
 * @example
 * deromanize('CCXIV');
 * // returns 214
 */
export const deromanize = (romanNum: string): number => {
  if (typeof romanNum !== 'string') {
    throw new Error('Input must be a string representing a Roman numeral');
  }

  let arabicNum = 0;
  romanNum = romanNum.toString().replace(/\s+/g, '').toUpperCase();

  const romanNumCharArray: string[] = Array.from(romanNum);

  romanNumCharArray.forEach((currentRomanNumChar: string, index: number) => {
    // search for invalid roman numerals
    const arabicValueForCurrentRomanNumChar: number | undefined =
      NUM_MAP.get(currentRomanNumChar);
    if (arabicValueForCurrentRomanNumChar === undefined) {
      throw new Error(`Invalid roman numeral: ${currentRomanNumChar}`);
    }

    if (index === romanNumCharArray.length - 1) {
      arabicNum += arabicValueForCurrentRomanNumChar;
    } else {
      const nextLowerArabicValue: number | undefined = NUM_MAP.get(
        romanNumCharArray[index + 1]
      );
      if (nextLowerArabicValue === undefined) {
        throw new Error(
          `Invalid roman numeral: ${romanNumCharArray[index + 1]}`
        );
      } else if (arabicValueForCurrentRomanNumChar < nextLowerArabicValue) {
        arabicNum -= arabicValueForCurrentRomanNumChar;
      } else {
        arabicNum += arabicValueForCurrentRomanNumChar;
      }
    }
  });

  if (romanize(arabicNum) !== romanNum) {
    throw new Error(`Invalid roman number: ${romanNum}`);
  }

  return arabicNum;
};
