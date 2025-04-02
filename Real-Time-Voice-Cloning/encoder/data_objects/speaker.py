from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance
from pathlib import Path

# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        
    def _load_utterances(self):
        # Read the _sources.txt file assuming each line is in the format "frames_filename|wav_filename"
        with self.root.joinpath("_sources.txt").open("r") as sources_file:
            lines = [line.strip() for line in sources_file if line.strip()]
            sources = {}
            for line in lines:
                tokens = line.split("|")
                if len(tokens) == 2:
                    sources[tokens[0]] = tokens[1]
                else:
                    print(f"Skipping malformed line: {line}")
        # Filter out entries where the wav file doesn't exist
        valid_sources = {}
        for frames_fname, wave_fname in sources.items():
            wav_path = self.root.joinpath(wave_fname)
            if wav_path.exists():
                valid_sources[frames_fname] = wave_fname
            else:
                print(f"Skipping missing file: {wav_path}")

        if not valid_sources:
            raise Exception("No valid utterances found in " + str(self.root))
        
        # Create utterance objects from valid entries
        self.utterances = [Utterance(self.root.joinpath(f), self.root.joinpath(w)) 
                           for f, w in valid_sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)

    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)
        partials = [(u,) + u.random_partial(n_frames) for u in utterances]
        return partials
