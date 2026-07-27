/**
 * Ambient type declarations for wa-sqlite (no bundled .d.ts).
 * Provides minimal types for the SQLite Worker build.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare module 'wa-sqlite/dist/wa-sqlite.mjs' {
  const factory: () => Promise<any>;
  export default factory;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare module 'wa-sqlite' {
  export const Factory: (module: any) => any;
}

declare module 'wa-sqlite/src/examples/AccessHandlePoolVFS.js' {
  // eslint-disable-next-line @typescript-eslint/no-extraneous-class
  class AccessHandlePoolVFS {
    constructor(name: string);
    name: string;
  }
  export { AccessHandlePoolVFS };
}
