// export_4channel.ijm — batch-export every series of .lif/.czi/.nd2 files as
// multi-channel TIFFs with all channels preserved (no RGB flattening).
//
// HOW TO RUN (Fiji):
//   Plugins > Macros > Run...  and pick this file.
//   You'll be asked for an input folder (the raw .lif/.czi files) and an
//   output folder for the TIFFs.
//
// WHY: exporting with Bio-Formats "RGB Color" mode merges channels into a
// 3-channel composite, which permanently mixes a far-red channel into R and B
// as magenta. This macro uses color_mode=Default, which keeps every channel
// separate at full bit depth.
//
// Output: <output>/<lifname>_<SeriesName>.tif  — a hyperstack whose channel
// count matches the acquisition. MMPS reads these directly.

inputDir  = getDirectory("Choose the folder with your RAW .lif / .czi / .nd2 files");
outputDir = getDirectory("Choose an OUTPUT folder for the multi-channel TIFFs");

setBatchMode(true);
run("Bio-Formats Macro Extensions");

list = getFileList(inputDir);
totalSaved = 0;

for (f = 0; f < list.length; f++) {
    name = list[f];
    lower = toLowerCase(name);
    if (!(endsWith(lower, ".lif") || endsWith(lower, ".czi")
          || endsWith(lower, ".nd2") || endsWith(lower, ".oib")
          || endsWith(lower, ".oif") || endsWith(lower, ".vsi"))) {
        continue;
    }
    path = inputDir + name;
    base = File.nameWithoutExtension;
    print("File: " + name);

    Ext.setId(path);
    Ext.getSeriesCount(seriesCount);
    print("   series: " + seriesCount);

    for (s = 0; s < seriesCount; s++) {
        Ext.setSeries(s);
        Ext.getSeriesName(seriesName);

        // color_mode=Default keeps channels SEPARATE (no RGB merge).
        run("Bio-Formats Importer",
            "open=[" + path + "]" +
            " color_mode=Default" +
            " view=Hyperstack" +
            " stack_order=XYCZT" +
            " series_" + (s + 1));

        getDimensions(w, h, chan, slices, frames);

        // Z-stack -> max projection (MMPS is 2D). Single slice passes through.
        if (slices > 1) {
            run("Z Project...", "projection=[Max Intensity]");
            // Z Project opens a new window; close the original stack
            proj = getTitle();
            selectWindow(proj);
        }

        // sanitize the series name for the filename
        safe = replace(seriesName, "[^A-Za-z0-9._-]", "_");
        outPath = outputDir + base + "_" + safe + ".tif";
        saveAs("Tiff", outPath);
        getDimensions(w2, h2, chan2, sl2, fr2);
        print("   [" + (s + 1) + "/" + seriesCount + "] " + safe
              + "  ->  " + chan2 + " channels, " + bitDepth() + "-bit");
        totalSaved++;
        close("*");
    }
    Ext.close();
}

setBatchMode(false);
print("");
print("Done. Saved " + totalSaved + " TIFFs to: " + outputDir);
print("Check one in MMPS: the BBB dialog should now list Channel 1..N.");
