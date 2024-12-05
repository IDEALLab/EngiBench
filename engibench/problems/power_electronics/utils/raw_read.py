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


def rawread(fname: str):
    """Read ngspice binary raw files. Return tuple of the data, and the
    plot metadata. The dtype of the data contains field names. This is
    not very robust yet, and only supports ngspice.
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
    plot = {}
    count = 0
    arrs = []
    plots = []
    while True:
        try:
            mdata = fp.readline(BSIZE_SP).split(b":", maxsplit=1)
        except:
            raise

        print("rawread.py: mdata:", mdata)

        if len(mdata) == 2:
            print("mdata len = 2", mdata)
            if mdata[0].lower() in MDATA_LIST:
                plot[mdata[0].lower()] = mdata[1].strip()
            if mdata[0].lower() == b"variables":
                nvars = int(plot[b"no. variables"])
                npoints = int(plot[b"no. points"])
                plot["varnames"] = []
                plot["varunits"] = []
                for varn in range(nvars):
                    varspec = fp.readline(BSIZE_SP).strip().decode("ascii").split()
                    assert varn == int(varspec[0])
                    plot["varnames"].append(varspec[1])
                    plot["varunits"].append(varspec[2])
            if mdata[0].lower() == b"binary":
                print("binary content:", mdata[1])
                rowdtype = np.dtype(
                    {
                        "names": plot["varnames"],
                        "formats": [np.complex_ if b"complex" in plot[b"flags"] else np.float_] * nvars,
                    }
                )
                # We should have all the metadata by now
                arrs.append(np.fromfile(fp, dtype=rowdtype, count=npoints))
                print("arrs", arrs)
                print("arrs[0].shape", arrs[0].shape)
                plots.append(plot)
                fp.readline()  # Read to the end of line
        else:
            print(f"length of mdata = {len(mdata)}")
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


# if __name__ == '__main__':
#    arrs, plots = rawread('./test/test.raw')
#    parse(arrs, plots)
