"""Read from .raw file created by ngSpice. Return variables for Efficiency calculation.
Adjusted from the original https://gist.github.com/snmishra/27dcc624b639c2626137?permalink_comment_id=4386356.
Modifications include adding comments and removing the parse() method, variables plots and other minor changes.

This script ONLY works for binary raw files that are written with set filetype=binary. ascii raw files will NOT work.
For an example of both binary and ascii raw files, see the end of this file.
For more details about how a raw file is structured, see https://github.com/nunobrum/PyLTSpice/blob/master/PyLTSpice/raw/raw_read.py.
"""

# ruff: noqa

from __future__ import annotations

from typing import Any

import numpy as np

BSIZE_SP = 512  # Max size of a line of data; we don't want to read the
# whole file to find a line, in case file does not have
# expected structure.
MDATA_LIST = [
    b"title",
    b"date",
    b"plotname",
    b"flags",
    b"no. variables",
    b"no. points",
    b"dimensions",
    b"command",
    b"option",
]


def rawread(fname: str) -> tuple[np.ndarray, dict]:
    """Read ngspice binary raw files. Return tuple of the data, and the plot metadata.
    The thing is that the .raw file contains both ascii/text and binary information(, assuming it's written using set filetype=binary).
    To accommodate this, we first use readline() for ascii texts, then use np.fromfile() to capture the values.

    In general, if you're dealing with a file that contains mixed content (both text and numbers), np.fromfile() may not be the best choice.
    Instead, you might want to use a different approach to read and parse the file,
    such as reading the file line by line, processing each line accordingly, and then converting the numeric data to a NumPy array.

    Note:   np.fromfile() does change the file pointer. When you read data from a file using np.fromfile(),
            the file pointer moves forward based on the amount of data read.e data that was read.
            This means that subsequent read operations will start from the new position of the file pointer.

            I played with the test files and tried to convince myself that the readline() at line 125 is necessary,
            by printing the remaining contents after running np.fromfile().
            However, I have not successfully understand what's remaining. Could be a todo in the future.

    The dtype of the data contains field names. This is not very robust yet, and only supports ngspice.
        >>> darr, mdata = rawread("test.py")
        >>> darr.dtype.names
        >>> plot(np.real(darr["frequency"]), np.abs(darr["v(out)"]))
    """
    # Example header of raw file
    # Title: rc band pass example circuit
    # Date: Sun Feb 21 11:29:14  2016
    # Plotname: AC Analysis
    # Flags: complex
    # No. Variables: 3
    # No. Points: 41
    # Variables:
    #         0       frequency       frequency       grid=3
    #         1       v(out)  voltage
    #         2       v(in)   voltage
    # Binary:
    fp = open(fname, "rb")
    plot: dict[Any, Any] = {}
    count = 0
    arrs = []
    plots = []
    while True:
        try:
            mdata = fp.readline(BSIZE_SP).split(b":", maxsplit=1)
        except:
            raise

        if len(mdata) == 2:
            # print("mdata len = 2", mdata)
            if mdata[0].lower() in MDATA_LIST:
                plot[mdata[0].lower()] = mdata[1].strip()
            if mdata[0].lower() == b"variables":
                nvars = int(plot[b"no. variables"])
                npoints = int(plot[b"no. points"])
                plot["varnames"] = []
                plot["varunits"] = []
                for varn in range(nvars):
                    varspec = fp.readline(BSIZE_SP).strip().decode("ascii").split()
                    vn, vname, vunit = varspec[0], varspec[1], varspec[2]
                    assert varn == int(vn)
                    """ Handle the following situation on Ubuntu:
                    Ubuntu:
                        ...
                        41 gs4 voltage
                        42 gs_ref_d voltage
                        ...

                    Windows (Expected):
                        ...
                        41 v(gs4) voltage
                        42 v(gs_ref_d) voltage
                        ...
                    """
                    if vunit == "voltage" and vname[0] != "v":
                        vname_ = f"v({vname})"
                        print(f"Changing {vname} to {vname_}.")
                    else:
                        vname_ = vname

                    plot["varnames"].append(vname_)  # "i(v_gs3)", "v(gs4)", "gain"
                    plot["varunits"].append(vunit)  # "current", "voltage", "notype"
                    """ An example of the variable {plot}:
                    {b'title': b'* rewrite netlist',
                    b'date': b'Tue Mar 18 19:55:46  2025',
                    b'plotname': b'Transient Analysis',
                    b'flags': b'real',
                    b'no. variables': b'59',
                    b'no. points': b'12001',
                    'varnames': ['time', 'i(@c0[i])', 'i(@c1[i])', 'i(@c2[i])', 'i(@c3[i])', 'i(@c4[i])', 'i(@c5[i])', 'i(@d0[i])', 'i(@d1[i])', 'i(@d2[i])', 'i(@d3[i])', 'i(@l0[i])', 'i(@l1[i])', 'i(@l2[i])', 'i(@r0[i])', 'i(@rc0[i])', 'i(@rc1[i])', 'i(@rc2[i])', 'i(@rc3[i])', 'i(@rc4[i])', 'i(@rc5[i])', 'i(@s0[i])', 'i(@s1[i])', 'i(@s2[i])', 'i(@s3[i])', 'i(@s4[i])', 'v(1)', 'v(2)', 'v(3)', 'v(4)', 'v(5)', 'v(6)', 'v(7)', 'v(8)', 'v(9)', 'v(10)', 'gain', 'v(gs0)', 'v(gs1)', 'v(gs2)', 'v(gs3)', 'v(gs4)', 'v(gs_ref_d)', 'v(gs_ref_dc)', 'i(l0)', 'i(l1)', 'i(l2)', 'i(v0)', 'i(v_gs0)', 'i(v_gs1)', 'i(v_gs2)', 'i(v_gs3)', 'i(v_gs4)', 'i(v_gsref_d)', 'i(v_gsref_dc)', 'vdiff', 'vo_mean', 'vpp', 'vpp_ratio'],
                    'varunits': ['time', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'voltage', 'voltage', 'voltage', 'voltage', 'voltage', 'voltage', 'voltage', 'voltage', 'voltage', 'voltage', 'notype', 'voltage', 'voltage', 'voltage', 'voltage', 'voltage', 'voltage', 'voltage', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'current', 'notype', 'notype', 'notype', 'notype']}"""

            if mdata[0].lower() == b"binary":
                # print("binary content:", mdata[1])
                """mdata = [b'Binary', b'\n'], len = 2 """
                names = plot["varnames"]
                formats = [np.complex_ if b"complex" in plot[b"flags"] else np.float64] * nvars

                rowdtype = np.dtype(
                    {"names": names, "formats": formats}
                )  # structured type dtype. https://numpy.org/doc/2.1/user/basics.rec.html
                """Expected rowdtype:
                [('time', '<f8'), ('i(@c0[i])', '<f8'), ('i(@c1[i])', '<f8'), ('i(@c2[i])', '<f8'), ('i(@c3[i])', '<f8'),
                ('i(@c4[i])', '<f8'), ('i(@c5[i])', '<f8'), ('i(@d0[i])', '<f8'), ('i(@d1[i])', '<f8'), ('i(@d2[i])', '<f8'),
                ('i(@d3[i])', '<f8'), ('i(@l0[i])', '<f8'), ('i(@l1[i])', '<f8'), ('i(@l2[i])', '<f8'), ('i(@r0[i])', '<f8'),
                ('i(@rc0[i])', '<f8'), ('i(@rc1[i])', '<f8'), ('i(@rc2[i])', '<f8'), ('i(@rc3[i])', '<f8'), ('i(@rc4[i])', '<f8'),
                ('i(@rc5[i])', '<f8'), ('i(@s0[i])', '<f8'), ('i(@s1[i])', '<f8'), ('i(@s2[i])', '<f8'), ('i(@s3[i])', '<f8'),
                ('i(@s4[i])', '<f8'), ('v(1)', '<f8'), ('v(2)', '<f8'), ('v(3)', '<f8'), ('v(4)', '<f8'), ('v(5)', '<f8'),
                ('v(6)', '<f8'), ('v(7)', '<f8'), ('v(8)', '<f8'), ('v(9)', '<f8'), ('v(10)', '<f8'), ('gain', '<f8'),
                ('v(gs0)', '<f8'), ('v(gs1)', '<f8'), ('v(gs2)', '<f8'), ('v(gs3)', '<f8'), ('v(gs4)', '<f8'), ('v(gs_ref_d)', '<f8'),
                ('v(gs_ref_dc)', '<f8'), ('i(l0)', '<f8'), ('i(l1)', '<f8'), ('i(l2)', '<f8'), ('i(v0)', '<f8'), ('i(v_gs0)', '<f8'),
                ('i(v_gs1)', '<f8'), ('i(v_gs2)', '<f8'), ('i(v_gs3)', '<f8'), ('i(v_gs4)', '<f8'), ('i(v_gsref_d)', '<f8'),
                ('i(v_gsref_dc)', '<f8'), ('vdiff', '<f8'), ('vo_mean', '<f8'), ('vpp', '<f8'), ('vpp_ratio', '<f8')]
                """
                # We should have all the metadata by now
                arrs.append(np.fromfile(fp, dtype=rowdtype, count=npoints))
                # print(f"Retrieved the values. np.fromfile() changed the pointer to {fp.tell()}.")
                # print("arrs", arrs)  # This looks better in jupyter notebook.
                # print("arrs[0].shape", arrs[0].shape)
                plots.append(plot)

                fp.readline()  # Move the pointer to the end of line.

        else:
            # print(f"length of mdata = {len(mdata)}. Break the iteration.")
            break

    return (arrs[0], plots[0])


def parse(arrs, plots):
    data_arr = {}
    num_var = len(plots["varnames"])
    process_arr = [[] for i in range(num_var)]

    for i in arrs:
        for j in range(len(i)):
            process_arr[j].append(i[j])

    var_names = plots["varnames"]
    for i in range(num_var):
        data_arr[var_names[i]] = np.array(process_arr[i])

    return data_arr


def process_time(time_arr, sim_start):
    for i in range(len(time_arr)):
        time_arr[i] -= sim_start

    return time_arr


if __name__ == "__main__":
    arrs, plots = rawread("./5_4_3_6_10-dcdc_converter_1_binary.raw")
    # parse(arrs, plots)
