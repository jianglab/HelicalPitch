import numpy as np
import pandas as pd
import os, pathlib

from joblib import Memory

username = os.getenv("USER")
memory = Memory(location=f"/tmp/{username}_joblib_cache", verbose=0)

from shiny import reactive
from shiny.express import ui, render, module, expressify


def compute_pair_distances(helices, lengths=None, target_total_count=-1):
    if lengths is not None:
        sorted_indices = (np.argsort(lengths))[::-1]
    else:
        sorted_indices = range(len(helices))
    min_len = 0
    dists_same_class = []
    for i in sorted_indices:
        _, segments_all_classes = helices[i]
        class_ids = np.unique(segments_all_classes["rlnClassNumber"])
        for ci in class_ids:
            mask = segments_all_classes["rlnClassNumber"] == ci
            segments = segments_all_classes.loc[mask, :]
            pos_along_helix = segments["rlnHelicalTrackLengthAngst"].values.astype(
                float
            )
            psi = segments["rlnAnglePsi"].values.astype(float)

            distances = np.abs(pos_along_helix[:, None] - pos_along_helix)
            distances = np.triu(distances)

            # Calculate pairwise distances only for segments with the same polarity
            mask = np.abs((psi[:, None] - psi + 180) % 360 - 180) < 90
            distances = distances[mask]
            dists_same_class.extend(
                distances[distances > 0]
            )  # Exclude zero distances (self-distances)
        if (
            lengths is not None
            and target_total_count > 0
            and len(dists_same_class) > target_total_count
        ):
            min_len = lengths[i]
            break
    if not dists_same_class:
        return [], 0
    else:
        return np.sort(dists_same_class), min_len

def find_first_peak(distances, cutoff=500):
    import numpy as np
    from scipy.stats import gaussian_kde
    distances_work = np.sort(np.hstack((-distances, distances)))
    kde = gaussian_kde(distances_work)
    kde_x = np.arange(np.min(distances_work).round(), np.max(distances_work).round()+0.5)
    kde_y = kde.pdf(kde_x)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(kde_y, prominence=(0.01/len(kde_x), None))
    peaks = np.sort(peaks)
    index_0 = (np.where(kde_x>=0))[0][0]
    peaks = peaks[peaks>=index_0] - index_0
    kde_x = kde_x[index_0:]
    kde_y = kde_y[index_0:]
    first_peak = 0
    if peaks is not None:
        if len(peaks)==1:
            if peaks[0]>=cutoff:
                first_peak=peaks[0]
        else:
            if peaks[0]>=cutoff:
                first_peak=peaks[0]
            else:
                if peaks[1]>=cutoff:
                    first_peak=peaks[1]
    return first_peak,peaks

def find_pitch(distances):
    import numpy as np
    from scipy.stats import gaussian_kde
    distances_work = np.sort(np.hstack((-distances, distances)))
    kde = gaussian_kde(distances_work)
    kde_x = np.arange(np.min(distances_work).round(), np.max(distances_work).round()+0.5)
    kde_y = kde.pdf(kde_x)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(kde_y, prominence=(0.01/len(kde_x), None))
    peaks = np.sort(peaks)
    index_0 = (np.where(kde_x>=0))[0][0]
    peaks = peaks[peaks>=index_0] - index_0
    kde_x = kde_x[index_0:]
    kde_y = kde_y[index_0:]
    print(peaks)
    if len(peaks)<3: return None
    dist = np.sort(peaks[1:]-peaks[:-1])
    pitch_tmp = np.percentile(dist, 50, method='lower')
    ns = peaks/pitch_tmp
    ns_int = np.round(ns)
    err = np.abs(ns-ns_int)
    peak_mask = np.where(err<0.1)
    if np.count_nonzero(peak_mask)<2: return None
    pitch = np.sum(peaks[peak_mask])/np.sum(ns_int[peak_mask])
    return (pitch, kde_x, kde_y)


def select_helices_by_length(helices, lengths, min_len, max_len):
    min_len = 0 if min_len is None else min_len
    max_len = -1 if min_len is None else max_len

    helices_retained = []
    n_ptcls = 0
    for gi, (gn, g) in enumerate(helices):
        cond = max_len <= 0 and min_len <= lengths[gi]
        cond = cond or (
            max_len > 0 and (max_len > min_len and (min_len <= lengths[gi] < max_len))
        )
        if cond:
            n_ptcls += len(g)
            helices_retained.append((gn, g))
    return helices_retained, n_ptcls


def get_filament_length(helices, particle_box_length=0):
    filement_lengths = []
    for gn, g in helices:
        track_lengths = g["rlnHelicalTrackLengthAngst"].astype(float).values
        length = track_lengths.max() - track_lengths.min() + particle_box_length
        filement_lengths.append(length)
    return filement_lengths


def select_classes(params, class_indices):
    class_indices_tmp = np.array(class_indices) + 1
    mask = params["rlnClassNumber"].astype(int).isin(class_indices_tmp)
    particles = params.loc[mask, :]
    helices = list(particles.groupby(["rlnMicrographName", "rlnHelicalTubeID"]))
    return helices

def select_helices_from_helixID(params, ids):
    mask = params["helixID"].astype(int).isin(ids)
    particles = params.loc[mask, :]
    helices = list(particles.groupby(["rlnMicrographName", "rlnHelicalTubeID"]))
    return helices


def get_class_abundance(params, nClass):
    abundance = np.zeros(nClass, dtype=int)
    for gn, g in params.groupby("rlnClassNumber"):
        abundance[int(gn) - 1] = len(g)
    return abundance


def get_number_helices_classes(params):
    nHelices = len(list(params.groupby(["rlnMicrographName", "rlnHelicalTubeID"])))
    nClasses = len(params["rlnClassNumber"].unique())
    return nHelices, nClasses


def get_pixel_size(
    data,
    attrs=[
        "micrograph_blob/psize_A",
        "rlnMicrographPixelSize",
        "rlnMicrographOriginalPixelSize",
        "blob/psize_A",
        "rlnImagePixelSize",
    ],
    return_source=False,
):
    try:
        sources = [data.attrs["optics"]]
    except:
        sources = []
    sources += [data]
    for source in sources:
        for attr in attrs:
            if attr in source:
                if attr in ["rlnImageName", "rlnMicrographName"]:
                    import mrcfile, pathlib

                    folder = pathlib.Path(data["starFile"].iloc[0])
                    if folder.is_symlink():
                        folder = folder.readlink()
                    folder = folder.resolve().parent
                    filename = source[attr].iloc[0].split("@")[-1]
                    filename = str((folder / "../.." / filename).resolve())
                    with mrcfile.open(filename, header_only=True) as mrc:
                        apix = float(mrc.voxel_size.x)
                else:
                    apix = float(source[attr].iloc[0])
                if return_source:
                    return apix, attr
                else:
                    return apix
    return None


def assign_segment_id(data, inter_segment_distance):
    assert "rlnHelicalTrackLengthAngst" in data
    tmp = (
        data.loc[:, "rlnHelicalTrackLengthAngst"].astype(float) / inter_segment_distance
    )
    err = (tmp - tmp.round()).abs()
    if np.sum(np.where(err > 0.1)) > 0:
        print(
            f"WARNING: it appears that the helical segments were extracted with different inter-segment distances"
        )
    helical_segment_id = tmp.round().astype(int)
    return helical_segment_id


def estimate_inter_segment_distance(data):
    # data must have been sorted by micrograph, rlnHelicalTubeID, and rlnHelicalTrackLengthAngst
    helices = data.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False)

    import numpy as np

    dists_all = []
    for _, particles in helices:
        if len(particles) < 2:
            continue
        dists = np.sort(particles["rlnHelicalTrackLengthAngst"].astype(float).values)
        dists = dists[1:] - dists[:-1]
        dists_all.append(dists)
    dists_all = np.hstack(dists_all)
    dist_seg = np.median(dists_all)  # Angstrom
    return dist_seg


def get_class2d_from_uploaded_file(fileobj):
    import os, tempfile

    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        return get_class2d_from_file(temp.name)


@memory.cache
def get_class2d_from_url(url):
    url_final = get_direct_url(url)  # convert cloud drive indirect url to direct url
    fileobj = download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    data = get_class2d_from_file(fileobj.name)
    return data


def get_class2d_from_file(classFile):
    import mrcfile

    with mrcfile.open(classFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, round(apix, 4)



def get_class2d_helix_params_from_url(url):
    df = get_class2d_params_from_url(url)
    helices = df.groupby(["rlnMicrographName", "rlnHelicalTubeID"])
    for hi, (_, helix) in enumerate(helices):
        l = helix["rlnHelicalTrackLengthAngst"].astype(float).max().round()
        df.loc[helix.index, "length"] = l
        df.loc[helix.index, "helixID"] = hi + 1   
    return df

@memory.cache
def get_class2d_params_from_url(url):
    url_final = get_direct_url(url)  # convert cloud drive indirect url to direct url
    fileobj = download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    data = get_class2d_params_from_file(fileobj.name)
    return data


def get_class2d_helix_params_from_file(params_file):
    df = get_class2d_params_from_file(params_file)
    helices = df.groupby(["rlnMicrographName", "rlnHelicalTubeID"])
    for hi, (_, helix) in enumerate(helices):
        l = helix["rlnHelicalTrackLengthAngst"].astype(float).max().round()
        df.loc[helix.index, "length"] = l
        df.loc[helix.index, "helixID"] = hi + 1   
    return df

def get_class2d_params_from_file(params_file):
    if params_file.endswith(".star"):
        params = star_to_dataframe(params_file)
    elif params_file.endswith(".cs"):
        params = cs_to_dataframe(params_file)
    
    required_attrs = np.unique(
        "rlnImageName rlnHelicalTubeID rlnHelicalTrackLengthAngst rlnClassNumber rlnAnglePsi".split()
    )
    missing_attrs = [attr for attr in required_attrs if attr not in params]
    if missing_attrs:
        raise ValueError(f"ERROR: parameters {missing_attrs} are not available")
    return params


def star_to_dataframe(starFile):
    import starfile

    d = starfile.read(starFile, always_dict=True)
    assert (
        "optics" in d and "particles" in d
    ), f"ERROR: {starFile} has {' '.join(d.keys())} but optics and particles are expected"
    data = d["particles"]
    data.attrs["optics"] = d["optics"]
    data.attrs["starFile"] = starFile
    return data


def cs_to_dataframe(cs_file):
    cs = np.load(cs_file)
    data = pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
    required_attrs = "blob/idx blob/path filament/filament_uid filament/arc_length_A alignments2D/class alignments2D/pose".split()
    missing_attrs = [attr for attr in required_attrs if attr not in data]
    if missing_attrs:
        msg = f"ERROR: required attrs '{', '.join(missing_attrs)}' are not included in {cs_file}"
        msg += "\nIf the particles in this CryoSPARC job were imported from a RELION star file, you can use the following command to create a star file and load that star file to HelicalPitch:\n"
        msg += "helicon images2star <this cs file> <output star file> --copyParm <original star file>"
        raise ValueError(msg)
    ret = pd.DataFrame()
    ret["rlnImageName"] = (
        (data["blob/idx"].astype(int) + 1).map("{:06d}".format)
        + "@"
        + data["blob/path"].str.decode("utf-8")
    )
    if "micrograph_blob/path" in data:
        ret["rlnMicrographName"] = data["micrograph_blob/path"]
    else:
        ret["rlnMicrographName"] = data["blob/path"].str.decode("utf-8")

    if data["filament/filament_uid"].min() > 1000:
        micrographs = data.groupby(["blob/path"])
        for _, m in micrographs:
            mapping = {
                v: i + 1
                for i, v in enumerate(sorted(m["filament/filament_uid"].unique()))
            }
            ret.loc[m.index, "rlnHelicalTubeID"] = m["filament/filament_uid"].map(
                mapping
            )
    else:
        ret["rlnHelicalTubeID"] = data["filament/filament_uid"].astype(int)

    ret["rlnHelicalTrackLengthAngst"] = (
        data["filament/arc_length_A"].astype(np.float32).values.round(2)
    )

    # 2D class assignments
    ret["rlnClassNumber"] = data["alignments2D/class"].astype(int) + 1
    ret["rlnAnglePsi"] = -np.rad2deg(data["alignments2D/pose"]).round(2)
    return ret


def download_file_from_url(url):
    import tempfile
    import requests
    import os

    if pathlib.Path(url).is_file():
        return open(url, "rb")
    try:
        filesize = get_file_size(url)
        local_filename = url.split("/")[-1]
        suffix = "." + local_filename
        fileobj = tempfile.NamedTemporaryFile(suffix=suffix)
        with requests.get(url) as r:
            r.raise_for_status()  # Check for request success
            fileobj.write(r.content)
        return fileobj
    except requests.exceptions.RequestException as e:
        print(e)
        return None


def get_direct_url(url):
    import re

    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1") != -1:
            return url
        elif url.find("dl=0") != -1:
            return url.replace("dl=0", "dl=1")
        else:
            return url + "?dl=1"
    elif url.find("sharepoint.com") != -1 and url.find("guestaccess.aspx") != -1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64

        data_bytes64 = base64.b64encode(bytes(url, "utf-8"))
        data_bytes64_String = (
            data_bytes64.decode("utf-8").replace("/", "_").replace("+", "-").rstrip("=")
        )
        return (
            f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
        )
    else:
        return url


def get_file_size(url):
    import requests

    response = requests.head(url)
    if "Content-Length" in response.headers:
        file_size = int(response.headers["Content-Length"])
        return file_size
    else:
        return None


def plot_histogram(
    data,
    title,
    xlabel,
    ylabel,
    max_pair_dist=None,
    bins=50,
    log_y=True,
    show_pitch_twist={},
    multi_crosshair=False,
    fig=None,
):
    import plotly.graph_objects as go

    if max_pair_dist is not None and max_pair_dist > 0:
        data = [d for d in data if d <= max_pair_dist]

    hist, edges = np.histogram(data, bins=bins)
    hist_linear = hist
    if log_y:
        hist = np.log10(1 + hist)

    center = (edges[:-1] + edges[1:]) / 2

    hover_text = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        hover_info = f"{xlabel.replace(" (Å)", "")}: {center[i]:.0f} ({left:.0f}-{right:.0f})Å<br>{ylabel}: {hist_linear[i]}"
        if show_pitch_twist:
            rise = show_pitch_twist["rise"]
            csyms = show_pitch_twist["csyms"]
            for csym in csyms:
                twist = 360 / (center[i] * csym / rise)
                hover_info += f"<br>Twist for C{csym}: {twist:.2f}°"
        hover_text.append(hover_info)

    if fig:
        fig.data[0].x = center
        fig.data[0].y = hist
        fig.data[0].text = hover_text
        fig.layout.title.text = title
    else:
        fig = go.FigureWidget()

        histogram = go.Bar(
            x=center,
            y=hist,
            name="Histogram",
            marker_color="blue",
            hoverinfo="none",
        )

        fig.add_trace(histogram)

        fig.data[0].text = hover_text
        fig.data[0].hoverinfo = "text"
        fig.update_layout(
            template="plotly_white",
            title_text=title,
            title_x=0.5,
            title_font=dict(size=12),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            autosize=True,
            hovermode="closest",
            hoverlabel=dict(bgcolor="white", font_size=12),
        )

        if multi_crosshair:
            for i in range(20):
                fig.add_vline(
                    x=0,
                    line_width=3 if i == 0 else 2,
                    line_dash="solid" if i == 0 else "dash",
                    line_color="green",
                    visible=False,
                )

            def update_vline(trace, points, state):
                if points.point_inds:
                    hover_x = points.xs[0]
                    with fig.batch_update():
                        for i, vline in enumerate(fig.layout.shapes):
                            x = hover_x * (i + 1)
                            vline.x0 = x
                            vline.x1 = x
                            if x <= fig.data[0].x.max():
                                vline.visible = True
                            else:
                                vline.visible = False

            fig.data[0].on_hover(update_vline)

    return fig
