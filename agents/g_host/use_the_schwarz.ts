export declare const getNonce: () => string | undefined;
/**
 * Signs the style tag with a base64-encoded string (nonce) to conforms to Content Security Policies.
 * This function has to be invoked before any picker is rendered if you aren't using Webpack for CSP.
 */
export declare const setNonce: (hash: string) => void;
