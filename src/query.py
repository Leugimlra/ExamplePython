import sys
import gzip
import json
import math


def main(input_file: str, queries_file: str, output_file: str):
    """
    Creates an index object and runs the queries method.
    This puts the query results into output_file.
    """
    term_index = index(input_file)
    term_index.run_queries(queries_file, output_file)


class index:
    def __init__(self, input_file: str):
        self.indexer(input_file)
        self.vocabualry: list = list(self.inv_index.keys())

    def indexer(self, input_file: bytes) -> dict:
        """
        inputFile:  JSON file containing preprocessed text
                    in .gz format

        returns:    invertedIndex = Dict(term : List(docID, position))
        """
        self.scene_list = self.load_JSON(input_file)
        self.doc_index = self.create_doc_index(self.scene_list)
        self.inv_index = self.create_inv_index(self.doc_index)
        self.collection_size = len(self.doc_index)
        return self.inv_index

    def create_inv_index(self, doc_index: dict) -> dict:
        """
        doc_index = List(Dict(doc_id : Dict(term: frequency))):
                    Index of documents and the terms+freq in that document

        returns:    invertedIndex = Dict(term : List(docID, frequency))
        """
        inv_index = {}
        doc_keys = list(doc_index.keys())
        for doc_key in doc_keys:
            posting = doc_index.get(doc_key)
            words = posting.keys()
            for word in words:
                if inv_index.get(word) is None:
                    inv_index[word] = {}
                inv_index[word][doc_key] = posting.get(word)
        return inv_index

    def create_doc_index(self, scenes: list) -> list:
        """
        returns:    index = List(Dict(doc_id : Dict(term: frequency)))
        """
        scene_index = {}
        for scene in scenes:
            text = scene.get("text").split(" ")
            doc_id = scene.get("sceneId")
            size = 0
            # traverse text and insert in index
            for i in range(len(text)):
                if scene_index.get(doc_id) is None:
                    scene_index[doc_id] = {}

                if text[i] == "":
                    continue
                # If word not included, add to dict, else increment its freq
                if scene_index.get(doc_id).get(text[i]) is None:
                    scene_index.get(doc_id)[text[i]] = 1
                else:
                    scene_index.get(doc_id)[text[i]] = (
                        scene_index.get(doc_id).get(text[i]) + 1
                    )
                size += 1
            scene_index.get(doc_id)["scene_size"] = size
        return scene_index

    def load_JSON(self, input_file: str) -> list:
        with gzip.open(input_file, "rb") as f:
            data = json.load(f)
        return data.get("corpus")

    def run_queries(self, query_file: str, output_file: str):
        """
        Writes results of queries to
        /output_file

        query_file: file containing queries
        """
        queries = []
        with open(query_file, "r") as f:
            for line in f:
                queries.append(line.split("\t"))

        with open(output_file, "w") as f:
            for query in queries:
                results = self.find_results(query)
                i = 1
                for r in results:
                    f.write(query[0] + "\t")
                    f.write(r[0] + "\t")
                    f.write(str(i) + "\t")
                    f.write(str(r[1]) + "\t")
                    f.write("\n")
                    i += 1

    def find_results(self, query: list) -> list:
        """
        Returns list([document, score])
        """
        mode = query[2]  # BM25 / QL
        words = query[3:]
        doc_scores = []

        if mode.lower() == "bm25":
            doc_scores = self.calc_bm25(words)
        if mode.lower() == "ql":
            doc_scores = self.calc_ql(words)

        doc_scores.sort(key=lambda x: x[1], reverse=True)  # sort by score
        doc_scores = doc_scores[:100]  # take top 100
        return doc_scores

    def find_docs_with_words(self, query: str):
        """
        Returns documents to iterate through and
        term document frequency
        """
        doc_set = {}
        term_freq = {}
        for term in query:
            term = term.strip("\n\t")
            docs = list(self.inv_index.get(term).keys())
            term_freq[term] = len(docs)
            if len(doc_set) == 0:
                doc_set = set(doc_set)

            for doc in docs:
                doc_set.add(doc)
        return list(doc_set), term_freq

    def calc_bm25(self, query: list) -> list:
        """
        Returns list([document, score])
        with BM25 scoring algorithm as described on pg250 in
        "Search Engines: Information Retrieval in Practice"
        by Croft, Metzler, and Strohman.
        """
        # collect term frequency within each query
        q_term_freqs = {}
        for t in query:
            t = t.strip("\n\t")
            if q_term_freqs.get(t) is None:
                q_term_freqs[t] = 1
            else:
                q_term_freqs[t] += 1

        doc_score = []
        ave_doc_length = self.calc_scene_lengths()[0]
        docs, doc_freq = self.find_docs_with_words(query)
        N = self.collection_size
        for doc in docs:
            score = 0
            for term in query:
                term = term.strip("\n\t")
                cur_doc_length = self.doc_index.get(doc).get("scene_size")
                K = 1.8 * (0.25 + ((0.75 * cur_doc_length) / ave_doc_length))
                ni = doc_freq.get(term)
                qfi = q_term_freqs.get(term)
                # catch if term does not occur
                fi = self.inv_index.get(term).get(doc)
                if fi is None:
                    fi = 0

                # No relevance information, so R and r_i == 0
                score += (
                    math.log(1 / ((ni + 0.5) / (N - ni + 0.5)))
                    * ((2.8 * fi) / (K + fi))
                    * ((6 * qfi) / (5 + qfi))
                )
            doc_score.append([doc, score])
        return doc_score

    def calc_ql(self, query: list) -> list:
        """
        Returns list([document, score])
        with Query Liklihood scoring algorithm as described on pg259 in
        "Search Engines: Information Retrieval in Practice"
        by Croft, Metzler, and Strohman.
        """
        doc_score = []
        docs, doc_freq = self.find_docs_with_words(query)
        mu = 300
        C = self.calc_scene_lengths()[1]
        for doc in docs:
            score = 0
            for term in query:
                term = term.strip("\n\t")
                cqi = self.get_collection_freq(term)
                D = self.doc_index.get(doc).get("scene_size")
                fqi = self.inv_index.get(term).get(doc)
                if fqi is None:
                    fqi = 0

                score += math.log((fqi + mu * (cqi / C)) / (D + mu))
            doc_score.append([doc, score])
        return doc_score

    def get_collection_freq(self, term: str) -> int:
        sum = 0
        for doc in self.inv_index.get(term).keys():
            sum += self.inv_index.get(term).get(doc)
        return sum

    def calc_scene_lengths(self) -> list:
        """
        returns:    [ave. scene len, total words, longest scene, smallest scene]
        """
        scene_lengths = self.inv_index.get("scene_size")
        max = float("-inf")
        max_scene = ""
        min = float("inf")
        min_scene = ""
        ave = 0
        count = 0
        total = 0
        for scene in scene_lengths:
            size = scene_lengths.get(scene)
            if size > max:
                max = size
                max_scene = scene[0]
            if size < min:
                min = size
                min_scene = scene[0]
            ave += size
            count += 1

        total = ave
        ave = ave / count
        return [ave, total, max_scene, min_scene]


if __name__ == "__main__":
    # Read arguments from command line, or use sane defaults for IDE.
    argv_len = len(sys.argv)
    input_file = sys.argv[1] if argv_len >= 2 else "shakespeare-scenes.json.gz"
    queries_file = sys.argv[2] if argv_len >= 3 else "trainQueries.tsv"
    output_file = sys.argv[3] if argv_len >= 4 else "trainQueries.results"

    main(input_file, queries_file, output_file)
