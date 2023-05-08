import shutil
import subprocess

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq, reverse_complement
from fire import Fire
from tqdm import tqdm

from patterncode.config import *
from patterncode.seq_utils import expand_iupac, pack_string
from patterncode.utils import cached_func, read_human_genome, Computation


class GenomeIndex(Computation):
    sequences = None

    def __init__(self, limit_num_seq, limit_seq_len):
        super().__init__()
        self.limit_num_seq = limit_num_seq
        self.limit_seq_len = limit_seq_len
        self._compute()

    @staticmethod
    def get_genome(name):
        print(f'Loading genome: {name}')
        kw = dict(
            limit_num_seq=LIMIT_NUM_SEQ,
            limit_seq_len=LIMIT_SEQ_LEN,
        )

        if name == RANDOM_GENOME:
            genome = cached_func(RandomGenome)(genome_len=RANDOM_GENOME_LEN, **kw)
        elif name == HUMAN_GENOME:
            genome = cached_func(NCBIGenome)(DATASET_HUMAN, **kw)
        elif name == BACTERIAL_GENOMES:
            genome = cached_func(NCBIGenome)(DATASET_BACTERIA, **kw)
        else:
            raise ValueError(f'Unknown genome name: {name}')
        genome.genome_name = name
        return genome

    def get_pattern_positions(self, pattern: str, add_rev_comp: bool = True):
        pos_lists = []
        seqs = expand_iupac(pattern)
        for seq in seqs:
            pos_lists.append(self.get_subseq_positions(seq))
            if add_rev_comp:
                reversed_seq = reverse_complement(seq)
                if seq != reversed_seq:
                    pos_lists.append(self.get_subseq_positions(reversed_seq))
        pos = np.sort(np.concatenate(pos_lists))
        return pos

    def get_subseq_positions(self, subseq: str) -> np.ndarray:
        packed = pack_string(subseq.encode(), len(subseq)).item()
        pos = [group.get(packed, []) for group in self.grouped]
        offsets = np.cumsum([0] + self.sequence_lens[:-1])
        assert len(pos) == len(offsets)
        return np.concatenate([p + i for p, i in zip(pos, offsets)]).astype(np.int64)

    def _compute(self):
        self.sequences = [
            seq[:self.limit_seq_len]
            for seq in self.sequences[:self.limit_num_seq]
        ]
        self.sequence_lens = [len(seq) for seq in self.sequences]
        self.genome_len = sum(self.sequence_lens)
        self.grouped = list(tqdm(map(self._group_by_subseq, self.sequences), desc='Packing sequences'))

    @staticmethod
    def _group_by_subseq(seq: Seq):
        seq_bytes = str(seq).encode()
        seq_bytes = seq_bytes.upper()
        packed = pack_string(seq_bytes, PATTERN_LEN)
        return pd.Series(packed).groupby(packed).groups


class RandomGenome(GenomeIndex):
    def __init__(self, genome_len, **kwargs):
        self.genome_len = int(genome_len)
        self.sequences = [
            Seq(''.join(np.random.choice(list(ACGT), size=self.genome_len)))
        ]
        super().__init__(**kwargs)


class NCBIGenome(GenomeIndex):
    def __init__(self, dataset_name, **kwargs):
        self.genome_data_dir = None
        self.summary_file = None
        self.seq_df = None
        self.dataset_name = dataset_name
        self.read_genome()
        super().__init__(**kwargs)

    def read_genome(self):
        if self.dataset_name == 'bacteria':
            self.genome_data_dir = UNPACK_DIR_BACTERIA
            self._read_bacterial_genome()
        elif self.dataset_name == 'human':
            self._read_human_genome()

    def _read_human_genome(self):
        self.sequences = cached_func(read_human_genome)()

    def _read_bacterial_genome(self):
        df = self.get_metadata_df()
        self.metadata_df = df
        seq_df = self.get_sequences_df(df)
        self.sequences = seq_df['seq'].tolist()
        self.seq_df = seq_df

    def _get_fasta_file(self, accession_id):
        files = list((self.genome_data_dir / accession_id).glob('*.fna'))
        print(files)
        assert len(files) == 1

        file = list(files)[0]
        assert file.exists()

        return file.relative_to(self.genome_data_dir)

    @staticmethod
    def get_metadata_df():
        dfs = []
        print(UNPACK_DIR_BACTERIA)
        for assembly_dir in Path(UNPACK_DIR_BACTERIA).iterdir():
            print(f"Reading '{assembly_dir}'")
            if not assembly_dir.name.startswith('.'):
                jsonl_file = assembly_dir / ASSEMBLY_SUMMARY_FILE
                print(f"'{jsonl_file}'")
                json_lines = pd.read_json(jsonl_file, lines=True, typ='series').to_list()
                df = pd.json_normalize(json_lines)
                df['fasta_file'] = list(assembly_dir.rglob('*.fna'))[0]
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        return df

    def get_sequences_df(self, data_df):
        return pd.DataFrame([
            {
                'seq_id': record.id,
                'seq_name': record.name,
                'seq_description': record.description,
                'seq_len': len(record.seq),
                'seq': record.seq,
                ORGANISM_TAX_ID: item[ORGANISM_TAX_ID],
                ORGANISM_NAME: item[ORGANISM_NAME],
                ACCESSION: item[ACCESSION],
            } | dict(item)
            for _, item in tqdm(data_df.T.items(), desc='reading fasta files')
            for record in SeqIO.parse(self.genome_data_dir / item['fasta_file'], 'fasta')
            if record.id.startswith('NC') and 'plasmid' not in record.description
        ])

    @classmethod
    def download_bacteria(cls):
        names = get_species_names(BACTERIA_NAMES)
        download_dir = Path(DOWNLOAD_DIR)
        download_dir.mkdir(exist_ok=True, parents=True)

        print('Downloading genomes')
        for name in names:
            file = Path(download_dir) / f'{name}.zip'
            if not file.exists():
                print('Downloading: ', name)
                cmd = [
                    'datasets', 'download', 'genome', 'taxon', f'"{name}"',
                    '--filename', str(file),
                    '--assembly-level', 'complete',
                    '--exclude-atypical',
                    '--reference',
                ]
                print(' '.join(cmd))
                subprocess.run(cmd, check=True)
            else:
                print('Skipping: ', name)
            unpack_dir = UNPACK_DIR_BACTERIA / name
            if not unpack_dir.exists():
                print('Unpacking: ', file, 'to', unpack_dir)
                unpack_dir.mkdir(exist_ok=True, parents=True)
                shutil.unpack_archive(file, unpack_dir)
            else:
                print('Skipping: ', unpack_dir)

    @classmethod
    def download_human(cls):
        name = 'human'
        print('Downloading: ', name)
        file = Path(DOWNLOAD_DIR) / f'human.zip'
        cmd = [
            'datasets', 'download', 'genome', 'taxon', 'human',
            '--filename', str(file),
            '--reference',
        ]
        print(' '.join(cmd))
        subprocess.run(cmd)
        unpack_dir = UNPACK_DIR_HUMAN
        unpack_dir.mkdir(exist_ok=True, parents=True)
        print('Unpacking: ', file, 'to', unpack_dir)
        shutil.unpack_archive(file, unpack_dir)

    @classmethod
    def download_genomes(cls):
        cls.download_bacteria()
        cls.download_human()


if __name__ == '__main__':
    Fire(NCBIGenome)
