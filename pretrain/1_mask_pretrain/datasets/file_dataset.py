

import os
import torch
import pickle


class FileDataset:
    def __init__(self, file_path, selected_col_ids=None, dtypes=None, separator="\t", cached_index=False):
        self.file_path = file_path
        assert os.path.exists(self.file_path), "Error: The local datafile {} not exists!".format(self.file_path)
        
        self.separator = separator
        if selected_col_ids is None:
            # default to all fields
            self.selected_col_ids = list(
                range(len(open(self.file_path).readline().rstrip("\n").split(self.separator))))
        else:
            self.selected_col_ids = [int(col_id) for col_id in selected_col_ids]

        if dtypes is None:
            # default to str
            self.dtypes = [str for col_id in self.selected_col_ids]
        else:
            self.dtypes = [eval(col_dtype) for col_dtype in dtypes.split(",")]
            assert len(self.dtypes) == len(self.selected_col_ids)

        self.cached_index = cached_index
        self.data_cnt = 0

        try:
            self.slice_id = torch.distributed.get_rank()
            self.slice_count = torch.distributed.get_world_size()
        except Exception:
            self.slice_id = 0
            self.slice_count = 1


        self._init_seek_index()
        self._compute_start_pos_and_row_count()
        self._reader = None  # Defer reader initialization until needed

    def _init_seek_index(self):
        if self.cached_index:
            cache_path = "{}.index".format(self.file_path)
            assert os.path.exists(cache_path), "cache file {} not exists!".format(cache_path)
            self.total_row_count, self.lineid_to_offset = pickle.load(open(cache_path, "rb"))
            print("local datafile {} use cached row_count and line_idx-to-offset mapping".format(self.file_path))
        else:
            # make an iteration over the file to get row_count and line_idx-to-offset mapping
            fp = open(self.file_path, "r")
            print("local datafile {} begin to initialize row_count and line_idx-to-offset mapping".format(self.file_path))
            self.total_row_count = 0
            offset = 0
            self.lineid_to_offset = []
            for line in fp:
                self.lineid_to_offset.append(offset)
                self.total_row_count += 1
                offset += len(line.encode('utf-8'))
            fp.close()
        print("local datafile {} finished initializing row_count and line_idx-to-offset mapping".format(self.file_path))

    def _compute_start_pos_and_row_count(self):
        self.row_count = self.total_row_count // self.slice_count
        if self.slice_id < self.total_row_count - self.row_count * self.slice_count:
            self.row_count += 1
            self.start_pos = self.row_count * self.slice_id
        else:
            self.start_pos = self.row_count * self.slice_id + (self.total_row_count - self.row_count * self.slice_count)

    def _get_reader(self):
        fp = open(self.file_path, "r")
        fp.seek(self.lineid_to_offset[self.start_pos])
        return fp

    def _seek(self, offset=0):
        try:
            print("slice_id {} seek offset {}".format(self.slice_id, self.start_pos + offset))
            self._reader.seek(self.lineid_to_offset[self.start_pos + offset])
            self.data_cnt = offset
        except Exception:
            print("slice_id {} seek offset {}".format(self.slice_id, offset))
            self._reader.seek(self.lineid_to_offset[offset])
            self.data_cnt = offset

    def __len__(self):
        return self.total_row_count

    def __getitem__(self, index):
        if not hasattr(self, '_reader') or self._reader is None or self._reader.closed:
            self._reader = self._get_reader()
        if self.data_cnt == self.row_count:
            print("reach the end of datafile, start a new reader")
            self.data_cnt = 0
            self._reader = self._get_reader()
        column_l = self._reader.readline().rstrip("\n").split(self.separator)
        self.data_cnt += 1
        column_l = [dtype(column_l[col_id]) for col_id, dtype in zip(self.selected_col_ids, self.dtypes)]
        
        return column_l

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the file pointer from the state
        state['_reader'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize the file pointer
        self._compute_start_pos_and_row_count()
        self._reader = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.slice_id = worker_info.id
            self.slice_count = worker_info.num_workers
        else:
            self.slice_id = 0
            self.slice_count = 1
        self._compute_start_pos_and_row_count()
        self._reader = self._get_reader()
        self.data_cnt = 0
        return self

