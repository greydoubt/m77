export { blobLikeSchema, isBlobLike } from "./blobs.js";
export { catchUnrecognizedEnum } from "./enums.js";
export type { ClosedEnum, OpenEnum, Unrecognized } from "./enums.js";
export type { Result } from "./fp.js";
export type { PageIterator, Paginator } from "./operations.js";
export { createPageIterator } from "./operations.js";
export { RFCDate } from "./rfcdate.js";


const dateRE = /^\d{4}-\d{2}-\d{2}$/;

export class RFCDate {
  private serialized: string;

  /**
   * Creates a new RFCDate instance using today's date.
   */
  static today(): RFCDate {
    return new RFCDate(new Date());
  }

  /**
   * Creates a new RFCDate instance using the provided input.
   * If a string is used then in must be in the format YYYY-MM-DD.
   *
   * @param date A Date object or a date string in YYYY-MM-DD format
   * @example
   * new RFCDate("2022-01-01")
   * @example
   * new RFCDate(new Date())
   */
  constructor(date: Date | string) {
    if (typeof date === "string" && !dateRE.test(date)) {
      throw new RangeError(
        "RFCDate: date strings must be in the format YYYY-MM-DD: " + date,
      );
    }

    const value = new Date(date);
    if (isNaN(+value)) {
      throw new RangeError("RFCDate: invalid date provided: " + date);
    }

    this.serialized = value.toISOString().slice(0, "YYYY-MM-DD".length);
    if (!dateRE.test(this.serialized)) {
      throw new TypeError(
        `RFCDate: failed to build valid date with given value: ${date} serialized to ${this.serialized}`,
      );
    }
  }

  toJSON(): string {
    return this.toString();
  }

  toString(): string {
    return this.serialized;
  }
}

export type APICall =
  | {
      status: "complete";
      request: Request;
      response: Response;
    }
  | {
      status: "request-error";
      request: Request;
      response?: undefined;
    }
  | {
      status: "invalid";
      request?: undefined;
      response?: undefined;
    };

export class APIPromise<T> implements Promise<T> {
  readonly #promise: Promise<[T, APICall]>;
  readonly #unwrapped: Promise<T>;

  readonly [Symbol.toStringTag] = "APIPromise";

  constructor(p: [T, APICall] | Promise<[T, APICall]>) {
    this.#promise = p instanceof Promise ? p : Promise.resolve(p);
    this.#unwrapped =
      p instanceof Promise
        ? this.#promise.then(([value]) => value)
        : Promise.resolve(p[0]);
  }

  then<TResult1 = T, TResult2 = never>(
    onfulfilled?:
      | ((value: T) => TResult1 | PromiseLike<TResult1>)
      | null
      | undefined,
    onrejected?:
      | ((reason: any) => TResult2 | PromiseLike<TResult2>)
      | null
      | undefined,
  ): Promise<TResult1 | TResult2> {
    return this.#promise.then(
      onfulfilled ? ([value]) => onfulfilled(value) : void 0,
      onrejected,
    );
  }

  catch<TResult = never>(
    onrejected?:
      | ((reason: any) => TResult | PromiseLike<TResult>)
      | null
      | undefined,
  ): Promise<T | TResult> {
    return this.#unwrapped.catch(onrejected);
  }

  finally(onfinally?: (() => void) | null | undefined): Promise<T> {
    return this.#unwrapped.finally(onfinally);
  }

  $inspect(): Promise<[T, APICall]> {
    return this.#promise;
  }
}



import * as z from "zod";

export function constDateTime(
  val: string,
): z.ZodType<string, z.ZodTypeDef, unknown> {
  return z.custom<string>((v) => {
    return (
      typeof v === "string" && new Date(v).getTime() === new Date(val).getTime()
    );
  }, `Value must be equivelant to ${val}`);
}

export const blobLikeSchema: z.ZodType<Blob, z.ZodTypeDef, Blob> =
  z.custom<Blob>(isBlobLike, {
    message: "expected a Blob, File or Blob-like object",
    fatal: true,
  });



export function isReadableStream<T = Uint8Array>(
  val: unknown,
): val is ReadableStream<T> {
  if (typeof val !== "object" || val === null) {
    return false;
  }

  // Check for the presence of methods specific to ReadableStream
  const stream = val as ReadableStream<Uint8Array>;

  // ReadableStream has methods like getReader, cancel, and tee
  return (
    typeof stream.getReader === "function" &&
    typeof stream.cancel === "function" &&
    typeof stream.tee === "function"
  );
}

export function isBlobLike(val: unknown): val is Blob {
  if (val instanceof Blob) {
    return true;
  }

  if (typeof val !== "object" || val == null || !(Symbol.toStringTag in val)) {
    return false;
  }

  const name = val[Symbol.toStringTag];
  if (typeof name !== "string") {
    return false;
  }
  if (name !== "Blob" && name !== "File") {
    return false;
  }

  return "stream" in val && typeof val.stream === "function";
}



declare const __brand: unique symbol;
export type Unrecognized<T> = T & { [__brand]: "unrecognized" };

export function catchUnrecognizedEnum<T>(value: T): Unrecognized<T> {
  return value as Unrecognized<T>;
}

type Prettify<T> = { [K in keyof T]: T[K] } & {};
export type ClosedEnum<T> = T[keyof T];
export type OpenEnum<T> =
  | Prettify<T[keyof T]>
  | Unrecognized<T[keyof T] extends number ? number : string>;




import { Result } from "./fp.js";

export type Paginator<V> = () => Promise<V & { next: Paginator<V> }> | null;

export type PageIterator<V, PageState = unknown> = V & {
  next: Paginator<V>;
  [Symbol.asyncIterator]: () => AsyncIterableIterator<V>;
  "~next"?: PageState | undefined;
};

export function createPageIterator<V>(
  page: V & { next: Paginator<V> },
  halt: (v: V) => boolean,
): {
  [Symbol.asyncIterator]: () => AsyncIterableIterator<V>;
} {
  return {
    [Symbol.asyncIterator]: async function* paginator() {
      yield page;
      if (halt(page)) {
        return;
      }

      let p: typeof page | null = page;
      for (p = await p.next(); p != null; p = await p.next()) {
        yield p;
        if (halt(p)) {
          return;
        }
      }
    },
  };
}

/**
 * This utility create a special iterator that yields a single value and
 * terminates. It is useful in paginated SDK functions that have early return
 * paths when things go wrong.
 */
export function haltIterator<V extends object>(
  v: V,
): PageIterator<V, undefined> {
  return {
    ...v,
    next: () => null,
    [Symbol.asyncIterator]: async function* paginator() {
      yield v;
    },
  };
}

/**
 * Converts an async iterator of `Result<V, E>` into an async iterator of `V`.
 * When error results occur, the underlying error value is thrown.
 */
export async function unwrapResultIterator<V, PageState>(
  iteratorPromise: Promise<PageIterator<Result<V, unknown>, PageState>>,
): Promise<PageIterator<V, PageState>> {
  const resultIter = await iteratorPromise;

  if (!resultIter.ok) {
    throw resultIter.error;
  }

  return {
    ...resultIter.value,
    next: unwrapPaginator(resultIter.next),
    "~next": resultIter["~next"],
    [Symbol.asyncIterator]: async function* paginator() {
      for await (const page of resultIter) {
        if (!page.ok) {
          throw page.error;
        }
        yield page.value;
      }
    },
  };
}

function unwrapPaginator<V>(
  paginator: Paginator<Result<V, unknown>>,
): Paginator<V> {
  return () => {
    const nextResult = paginator();
    if (nextResult == null) {
      return null;
    }
    return nextResult.then((res) => {
      if (!res.ok) {
        throw res.error;
      }
      const out = {
        ...res.value,
        next: unwrapPaginator(res.next),
      };
      return out;
    });
  };
}

export const URL_OVERRIDE = Symbol("URL_OVERRIDE");
  
export type Result<T, E = unknown> =
  | { ok: true; value: T; error?: never }
  | { ok: false; value?: never; error: E };

export function OK<V>(value: V): Result<V, never> {
  return { ok: true, value };
}

export function ERR<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

/**
 * unwrap is a convenience function for extracting a value from a result or
 * throwing if there was an error.
 */
export function unwrap<T>(r: Result<T, unknown>): T {
  if (!r.ok) {
    throw r.error;
  }
  return r.value;
}

/**
 * unwrapAsync is a convenience function for resolving a value from a Promise
 * of a result or rejecting if an error occurred.
 */
export async function unwrapAsync<T>(
  pr: Promise<Result<T, unknown>>,
): Promise<T> {
  const r = await pr;
  if (!r.ok) {
    throw r.error;
  }

  return r.value;
}
