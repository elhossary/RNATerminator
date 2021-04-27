import pandas as pd
import os
import re
from io import StringIO
import logging as logger


class Wiggle:

    def __init__(self, file_path, chrom_sizes, is_len_extended):
        self.file_path = file_path
        self.chrom_sizes = chrom_sizes
        self.wiggle_df_columns = ["track_type", "track_name", "variableStep_chrom",
                                  "variableStep_span", "location", "score"]
        self.wiggle_df = pd.DataFrame(columns=self.wiggle_df_columns)
        self.wiggle_df["location"] = self.wiggle_df["location"].astype(int)
        self.orientation = None
        self.parse(is_len_extended)

    def parse(self, is_len_extended=False):
        logger.info(f"Parsing {self.file_path}")
        current_wiggle_meta = {}
        ignored_seqid = []
        with open(self.file_path, "r") as raw_file:
            print(f"==> Loading file: {os.path.basename(self.file_path)}")
            file_header, all_contents = self._parse_wiggle_str(raw_file.read())
            current_wiggle_meta = self.parse_wiggle_header(file_header, current_wiggle_meta)
            for content_header, content in all_contents.items():
                current_wiggle_meta = self.parse_wiggle_header(content_header, current_wiggle_meta)
                tmp_df = pd.read_csv(StringIO(content), sep=" ", names=["location", "score_new"],
                                     dtype={"location": int, "score": float})
                tmp_df["track_type"] = current_wiggle_meta["track_type"]
                tmp_df["track_name"] = current_wiggle_meta["track_name"]
                tmp_df["variableStep_chrom"] = current_wiggle_meta["variableStep_chrom"]
                tmp_df["variableStep_span"] = current_wiggle_meta["variableStep_span"]
                if is_len_extended:
                    chrom_size = 0
                    for chrom in self.chrom_sizes:
                        if current_wiggle_meta["variableStep_chrom"] == chrom["seqid"]:
                            chrom_size = chrom["size"]
                            break
                    if chrom_size > 0:
                        tmp_lst = [{"track_type": current_wiggle_meta["track_type"],
                                    "track_name": current_wiggle_meta["track_name"],
                                    'variableStep_chrom': current_wiggle_meta["variableStep_chrom"],
                                    "variableStep_span": current_wiggle_meta["variableStep_span"],
                                    "location": x} for x in range(1, chrom_size + 1, 1)]
                        append_df = pd.DataFrame(columns=self.wiggle_df_columns)
                        append_df = append_df.append(tmp_lst, ignore_index=True)
                        del tmp_lst
                        join_columns = ["track_type", "track_name", "variableStep_chrom",
                                        "variableStep_span", "location"]
                        append_df = pd.merge(how='left',
                                             left=append_df, right=tmp_df,
                                             left_on=join_columns, right_on=join_columns)
                        append_df["score"] = append_df["score"].combine_first(append_df["score_new"])
                        append_df.drop(["score_new"], inplace=True, axis=1)
                        self.wiggle_df = self.wiggle_df.append(append_df)
                        del append_df
                    else:
                        ignored_seqid.append(current_wiggle_meta["variableStep_chrom"])
                else:
                    tmp_df.rename(columns={"score_new": "score"}, inplace=True)
                    self.wiggle_df = self.wiggle_df.append(tmp_df)
                    del tmp_df
            self.wiggle_df["score"] = self.wiggle_df["score"].fillna(0.0)
            self.wiggle_df.reset_index(drop=True, inplace=True)
            self.wiggle_df["location"] = pd.to_numeric(self.wiggle_df["location"], downcast='integer')
            self.wiggle_df["score"] = pd.to_numeric(self.wiggle_df["score"], downcast='float')
            # Logging
            wiggle_seqid = self.wiggle_df["variableStep_chrom"].unique().tolist()
            condition_name = self.wiggle_df["track_name"].unique().tolist()
            s = '\n     └── '
            if len(ignored_seqid) == 0:
                print(f"===> Parsed condition: {', '.join(condition_name)}\n"
                      f"     + Included sequence IDs:\n"
                      f"     └── {s.join(wiggle_seqid)}")
            else:
                print(f"===> Parsed condition: {', '.join(condition_name)}\n"
                      f"     + Included sequence IDs:\n"
                      f"     └── {s.join(wiggle_seqid)}\n"
                      f"     + Ignored sequence IDs:\n"
                      f"     └── {s.join(ignored_seqid)}")
        if self.orientation is None:
            if any(self.wiggle_df["score"] < 0):
                self.orientation = "r"
            else:
                self.orientation = "f"


    @staticmethod
    def _parse_wiggle_str(in_str):
        ret_dict = {}
        header_text = in_str.split("\n", maxsplit=1)[0]
        in_str = in_str.replace(f"{header_text}\n", "")
        all_headers = re.findall(r'^.*chrom=.*$', in_str, flags=re.MULTILINE | re.IGNORECASE)
        splitters = ""
        for header in all_headers:
            splitters += header + "|"
        splitters = f"({splitters[:-1]})"
        split_str_list = re.split(rf"{splitters}", in_str, flags=re.MULTILINE | re.IGNORECASE)
        content_list = [i for i in split_str_list if i != '']
        for i in range(0, len(content_list), 2):
            ret_dict[content_list[i]] = content_list[i + 1]
        return header_text, ret_dict

    @staticmethod
    def parse_wiggle_header(line, current_wiggle_meta):
        if "type=" in line:
            current_wiggle_meta["track_type"] = line.split('type=')[-1].split(' ')[0].replace('\"', '')
        if "name=" in line:
            current_wiggle_meta["track_name"] = line.split('name=')[-1].replace('\n', '').replace('\"', '')
        if "chrom=" in line:
            current_wiggle_meta["variableStep_chrom"] = line.split('chrom=')[-1].split(' ')[0].replace('\"', '')
        if "span=" in line:
            current_wiggle_meta["variableStep_span"] = line.split('span=')[-1].replace('\n', '').replace('\"', '')
        return current_wiggle_meta

    def get_wiggle(self):
        #self.parse(is_len_extended=is_len_extended)
        return self.wiggle_df

    def to_step_height(self, step_range, step_direction, inplace=False):
        #self.parse(is_len_extended=True)
        cond_name = self.wiggle_df.iat[0, 1]
        if step_direction == "start_end":
            print(f"==> Transforming '{cond_name}' to rising step height")
        else:
            print(f"==> Transforming '{cond_name}' to falling step height")
        ret_df = self.wiggle_df.copy()
        seqids = ret_df["variableStep_chrom"].unique().tolist()
        for seqid in seqids:
            if self.orientation == "r":
                ret_df.loc[ret_df["variableStep_chrom"] == seqid, "score"] = \
                    self._generate_step_height_col(ret_df[ret_df["variableStep_chrom"] == seqid]["score"].abs(),
                                                   step_range, step_direction, self.orientation) * -1
            else:
                ret_df.loc[ret_df["variableStep_chrom"] == seqid, "score"] = \
                    self._generate_step_height_col(ret_df[ret_df["variableStep_chrom"] == seqid]["score"],
                                                   step_range, step_direction, self.orientation)
        if inplace:
            self.wiggle_df = ret_df
            del ret_df
        else:
            return ret_df

    @staticmethod
    def _generate_step_height_col(in_col, step_range, step_direction, orientation):
        df = pd.DataFrame()
        df["scores"] = in_col
        """
        if (orientation == "f" and step_direction == "start_end") or \
                (orientation == "r" and step_direction == "end_start"):
            df["step_height"] = df["scores"].diff()
        else:
            df["scores"] = df["scores"].values[::-1]
            df["step_height"] = df["scores"].diff()
            df["scores"] = df["scores"].values[::-1]
            df["step_height"] = df["step_height"].values[::-1]

        """
        df["mean_before"] = df["scores"]
        df["mean_after"] = df["scores"].shift(-(step_range + 1))
        df["mean_before"] = df["mean_before"].rolling(step_range).mean()
        df["mean_after"] = df["mean_after"].rolling(step_range).mean()
        if (orientation == "f" and step_direction == "start_end") or \
                (orientation == "r" and step_direction == "end_start"):
            df["step_height"] = df["mean_after"] - df["mean_before"]
            df["step_height"] = df["step_height"].shift(1)
        else:
            df["step_height"] = df["mean_before"] - df["mean_after"]

        df[df["step_height"] < 0] = 0.0
        df.fillna(0.0, inplace=True)
        df.rename(columns={"step_height": in_col.name}, inplace=True)
        return df[in_col.name]