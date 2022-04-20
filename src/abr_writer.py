
import numpy as np

class ABRWriter:
    def __init__(self, dataset, waves, filename, freqs, spls, fileindex=None):
        """
        Write one ABR file for Brad Buran's analysis program, in "EPL"
        format.

        TODO : fix this so that the ABR-files have names and check that it works.

        ABR-click-time  indicate click intenstiry series run in this
            directory, use time to distinguish file
            Example: ABR-click-0901.abr
        ABR-tone-time-freq indicate tone run number in this directory
            with frequency in Hz (6 places).
            Example: ABR-tone-0945-005460.dat

        """
        self.freqs = freqs
        self.spls = spls
        twaves = waves.T
        abr_filename = "ABR-{0:s}-{1:4s}".format(dataset["stimtype"], filename[8:12])
        if dataset["stimtype"] == "tonepip":
            abr_filename = abr_filename + (
                "{0:06d}".format(dataset["Freqs"][fileindex])
            )
        abr_filename = abr_filename + ".dat"  # give name an extension
        np.savetxt(abr_filename, twaves, delimiter="\t ", newline="\r\n")
        header = self.make_header(filename, fileindex)
        self.rewriteheader(abr_filename, filename, header)
        return abr_filename

    def make_header(self, filename, fileindex):
        """
        Create a (partially faked) header string for Buran's ABR analysis program, using data from our
        files.
        This mimics the Eaton Peabody data header style.

        Parameters
        ----------
        filename : str
            Name of the file whose information from the freqs and spl dictionaries
            will be stored here
        fileindex : int
            Index into the dict for the specific frequency.

        Returns
        -------
        header : str
            the whole header, ready for insertion into the file.

        """

        header1 = ":RUN-3	LEVEL SWEEP	TEMP:207.44 20120427 8:03 AM	-3852.21HR: \n"
        header2 = ":SW EAR: R	SW FREQ: " + "%.2f" % self.freqs[filename][fileindex]
        header2 += (
            "	# AVERAGES: 512	REP RATE (/sec): 40	DRIVER: Starship	SAMPLE (usec): 10 \n"
        )
        header3 = ":NOTES- \n"
        header4 = ":CHAMBER-412 \n"
        levs = self.spls[filename]  # spls[ABRfiles[i]]
        levels = ["%2d;".join(int(l)) for l in levs]
        header5 = ":LEVELS:" + levels + " \n"
        #            header5 = ":LEVELS:20;25;30;35;40;45;50;55;60;65;70;75;80;85;90; \n"
        header6 = ":DATA \n"
        header = header1 + header2 + header3 + header4 + header5 + header6
        return header

    def rewriteheader(self, filename, header):
        """
        Write the file header before the rest of the data, replacing any
        previous header.

        Parameters
        ----------
        filename : str
            name of the file to which the header will be prepended

        header : str
            The header string to prepend

        """
        ABRfile = open(filename, "r")

        # remove any existing 'header' from the file, in case there are duplicate header rows in the wrong places
        datalines = []
        for line in ABRfile:
            if line[0] == ":":
                continue
            datalines.append(line)
        ABRfile.close()

        # now rewrite the file, prepending the header above the data
        ABRfile = open(filename, "w")
        ABRfile.write("".join(header))
        ABRfile.write("\n".join(datalines))
        ABRfile.write("".join("\r\r"))
        ABRfile.close()