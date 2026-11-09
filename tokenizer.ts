// Word-level tokenizer with GPT-2 byte-level pre-tokenization, matching tokenizer.json
// (NFKC + lowercase, add_prefix_space, GPT-2 split regex). No Node or DOM APIs, so it
// is shared by model.ts (Node/Bun) and web/autocomplete.ts (browser).

function bytesToUnicode(): Map<number, string> {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n++);
    }
  }
  return new Map(bs.map((b, i) => [b, String.fromCodePoint(cs[i])]));
}

const BYTE_ENC = bytesToUnicode();
const BYTE_DEC = new Map([...BYTE_ENC].map(([b, c]) => [c, b]));
const GPT2_SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** Byte-level pre-tokens ("Ġword" style), exactly as the Python tokenizer splits text. */
export function preTokenize(text: string): string[] {
  let norm = text.normalize("NFKC").toLowerCase();
  if (norm && !/^\s/.test(norm)) norm = " " + norm; // ByteLevel add_prefix_space
  return (norm.match(GPT2_SPLIT) ?? []).map((piece) => [...encoder.encode(piece)].map((b) => BYTE_ENC.get(b)!).join(""));
}

/** Byte-level piece(s) back to text. */
export function decodePieces(pieces: string): string {
  return decoder.decode(new Uint8Array([...pieces].map((c) => BYTE_DEC.get(c) ?? 63)));
}

export class WordTokenizer {
  vocab: Map<string, number>;
  inverse: string[];
  special: Set<number>;
  unkId: number;
  bosId: number;

  constructor(pieces: string[], unkId: number, bosId: number, special: Iterable<number>) {
    this.unkId = unkId;
    this.bosId = bosId;
    this.inverse = pieces;
    this.vocab = new Map(pieces.map((p, i) => [p, i]));
    this.special = new Set(special);
  }

  /** tokenizer.json (WordLevel model, BOS template post-processor) -> tokenizer. */
  static fromJson(json: any): WordTokenizer {
    const pieces: string[] = [];
    for (const [tok, id] of Object.entries<number>(json.model.vocab)) pieces[id] = tok;
    const bos = json.post_processor?.single?.find((p: any) => p.SpecialToken)?.SpecialToken?.id;
    const special = (json.added_tokens ?? []).filter((t: any) => t.special).map((t: any) => t.id);
    return new WordTokenizer(pieces, json.model.vocab[json.model.unk_token], json.model.vocab[bos], special);
  }

  idOf(piece: string): number {
    return this.vocab.get(piece) ?? this.unkId;
  }

  encode(text: string): number[] {
    return [this.bosId, ...preTokenize(text).map((p) => this.idOf(p))];
  }

  decode(ids: number[]): string {
    return decodePieces(ids.filter((id) => !this.special.has(id)).map((id) => this.inverse[id]).join(""));
  }
}
