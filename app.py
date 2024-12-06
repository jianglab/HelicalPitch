import pathlib
import numpy as np

from shiny import reactive, req
from shiny.express import input, ui, render
from shinywidgets import render_plotly

import helicon

from . import compute

params = reactive.value(None)

data_all = reactive.value(None)
abundance = reactive.value([])
image_size = reactive.value(0)

displayed_class_ids = reactive.value([])
displayed_class_images = reactive.value([])
displayed_class_title = reactive.value("Select class(es):")
displayed_class_labels = reactive.value([])

initial_selected_image_indices = reactive.value([0])
selected_images = reactive.value([])
selected_image_labels = reactive.value([])

selected_helices = reactive.value(([], [], 0))
retained_helices_by_length = reactive.value([])
pair_distances = reactive.value([])


ui.head_content(ui.tags.title("HelicalPitch"))
helicon.shiny.google_analytics(id="G-998MGRETTF")
helicon.shiny.setup_ajdustable_sidebar()
helicon.shiny.set_input_text_numeric_update_on_change()
ui.tags.style(
    """
    * { font-size: 10pt; padding:0; border: 0; margin: 0; }
    aside {--_padding-icon: 10px;}
    """
)
urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_data.star",
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
    )
}
url_key = "empiar-10940_job010"

with ui.sidebar(
    width="33vw", style="display: flex; flex-direction: column; height: 100%;"
):
    with ui.div(id="input_files", style="flex-shrink: 0;"):
        ui.input_radio_buttons(
            "input_mode_params",
            "How to obtain the Class2D parameter file:",
            choices=["upload", "url"],
            selected="url",
            inline=True,
        )
        with ui.panel_conditional("input.input_mode_params === 'upload'"):
            ui.input_file(
                "upload_params",
                "Upload the class2d parameters in a RELION star or cryoSPARC cs file",
                accept=[".star", ".cs"],
                placeholder="star or cs file",
            )

        with ui.panel_conditional("input.input_mode_params === 'url'"):
            ui.input_text(
                "url_params",
                "Download URL for a RELION star or cryoSPARC cs file",
                value=urls[url_key][0],
            )

        ui.input_radio_buttons(
            "input_mode_classes",
            "How to obtain the class average images:",
            choices=["upload", "url"],
            selected="url",
            inline=True,
        )

        with ui.panel_conditional("input.input_mode_classes === 'upload'"):
            ui.input_file(
                "upload_classes",
                "Upload the class averages in MRC format (.mrcs, .mrc)",
                accept=[".mrcs", ".mrc"],
                placeholder="mrcs or mrc file",
            )

        with ui.panel_conditional("input.input_mode_classes === 'url'"):
            ui.input_text(
                "url_classes",
                "Download URL for a RELION or cryoSPARC Class2D output mrc(s) file",
                value=urls[url_key][1],
            )

        ui.input_task_button("run", label="Run", style="width: 100%;")

    with ui.div(id="class-selection", style="flex-grow: 1; overflow-y: auto;"):
        helicon.shiny.image_select(
            id="select_classes",
            label=displayed_class_title,
            images=displayed_class_images,
            image_labels=displayed_class_labels,
            image_size=reactive.value(128),
            initial_selected_indices=initial_selected_image_indices,
        )

        @reactive.effect
        @reactive.event(input.select_classes)
        def update_selected_images():
            selected_images.set(
                [displayed_class_images()[i] for i in input.select_classes()]
            )
            selected_image_labels.set(
                [displayed_class_labels()[i] for i in input.select_classes()]
            )


title = "HelicalPitch: determine helical pitch/twist using 2D Classification info"
ui.h1(title, style="font-weight: bold;")

with ui.layout_columns(col_widths=(5, 7, 12)):
    with ui.card():
        with ui.div(style="max-height: 40vh; overflow-y: auto;"):

            @render.ui
            @reactive.event(selected_images)
            def display_selected_images():
                return helicon.shiny.image_gallery(
                    id="display_selected_image",
                    label=reactive.value("Selected classe(s):"),
                    images=selected_images,
                    image_labels=selected_image_labels,
                )

        with ui.layout_columns(col_widths=[12, 12], style="align-items: flex-end;"):

            @render_plotly
            @reactive.event(selected_helices, input.bins)
            def lengths_histogram_display():
                req(input.bins() is not None and input.bins() > 0)
                fig = getattr(lengths_histogram_display, "fig", None)
                helices, lengths, count = selected_helices()
                data = lengths
                class_indices = [
                    str(displayed_class_ids()[i] + 1) for i in input.select_classes()
                ]
                title = f"Filament Lengths: Class {' '.join(class_indices)}<br><i>{len(helices):,} filaments | {count:,} segments</i>"
                xlabel = "Filament Legnth (Å)"
                ylabel = "# of Filaments"
                log_y = True
                nbins = input.bins()
                fig = compute.plot_histogram(
                    data=data,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    bins=nbins,
                    log_y=log_y,
                    fig=fig,
                )
                lengths_histogram_display.fig = fig

                return fig

            with ui.layout_columns(
                col_widths=[6, 6, 12], style="align-items: flex-end;"
            ):
                ui.input_numeric(
                    "min_len", "Minimal length (Å)", min=0.0, value=0, step=1.0
                )
                ui.input_numeric(
                    "rise",
                    "Helical rise (Å)",
                    min=0.01,
                    max=1000.0,
                    value=4.75,
                    step=0.01,
                )
                with ui.accordion(id="additional_parameters", open=False):
                    with ui.accordion_panel(title="Additional parameters:"):
                        with ui.layout_columns(
                            col_widths=6, style="align-items: flex-start;"
                        ):
                            ui.input_checkbox(
                                "ignore_blank", "Ignore blank classes", value=True
                            )
                            ui.input_checkbox(
                                "sort_abundance",
                                "Sort the classes by abundance",
                                value=True,
                            )
                            ui.input_checkbox(
                                "auto_min_len",
                                "Auto-set minimal filament length",
                                value=True,
                            )
                            ui.input_checkbox(
                                "show_sharable_url",
                                "Show sharable URL",
                                value=False,
                            )
                        with ui.layout_columns(
                            col_widths=6, style="align-items: flex-end;"
                        ):
                            ui.input_numeric(
                                "max_len",
                                "Maximal length (Å)",
                                min=-1,
                                value=-1,
                                step=1.0,
                            )
                            ui.input_numeric(
                                "max_pair_dist",
                                "Maximal pair distance (Å) to plot",
                                min=-1,
                                value=-1,
                                step=1.0,
                            )
                            ui.input_numeric(
                                "bins",
                                "Number of histogram bins",
                                min=1,
                                value=100,
                                step=1,
                            )

                @reactive.effect
                @reactive.event(input.min_len)
                def _():
                    ui.update_numeric("max_len", min=input.min_len())
                    if 0 < input.max_len() < input.min_len():
                        ui.update_numeric("max_len", value=-1)

                @reactive.effect
                @reactive.event(input.max_len)
                def _():
                    if input.max_len() > 0:
                        ui.update_numeric("min_len", max=input.max_len())
                    if input.min_len() >= input.max_len():
                        ui.update_numeric("min_len", value=0)

    with ui.card(max_height="90vh"):

        @render_plotly
        @reactive.event(pair_distances, input.bins, input.max_pair_dist, input.rise)
        def pair_distances_histogram_display():
            req(input.bins() is not None and input.bins() > 0)
            req(input.max_pair_dist() is not None)
            req(input.rise() is not None and input.rise() > 0)
            fig = getattr(pair_distances_histogram_display, "fig", None)
            data = pair_distances()
            segment_count = np.sum([len(h) for hi, h in retained_helices_by_length()])
            if len(retained_helices_by_length()):
                class_indices = np.unique(
                    np.concatenate(
                        [h["rlnClassNumber"] for hi, h in retained_helices_by_length()]
                    )
                ).astype(int)
            else:
                class_indices = []
            class_indices = [
                str(displayed_class_ids()[i] + 1)
                for i in input.select_classes()
                if (displayed_class_ids()[i] + 1) in class_indices
            ]
            rise = input.rise()
            log_y = True
            title = f"Pair Distances: Class {' '.join(class_indices)}<br><i>{len(retained_helices_by_length())} filaments | {segment_count:,} segments | {len(pair_distances()):,} segment pairs"
            xlabel = "Pair Distance (Å)"
            ylabel = "# of Pairs"
            nbins = input.bins()
            max_pair_dist = input.max_pair_dist()

            fig = compute.plot_histogram(
                data=data,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                max_pair_dist=max_pair_dist,
                bins=nbins,
                log_y=log_y,
                show_pitch_twist=dict(rise=rise, csyms=(1, 2, 3, 4)),
                multi_crosshair=True,
                fig=fig,
            )
            pair_distances_histogram_display.fig = fig

            return fig

        ui.markdown(
            "**How to interpret the histogram:** An informative histogram should have clear peaks with equal spacing. If so, hover your mouse pointer over the first major peak off the origin to align the vertical lines well with the peaks. Once you have decided on the line position, read the hover text, which shows the twist values assuming the pair distance is the helical pitch (adjusted for the cyclic symmetries around the helical axis). You need to decide which cyclic symmetry and the corresponding twist should be used.  \n&nbsp;&nbsp;&nbsp;&nbsp;If the histogram does not show clear peaks, it indicates that the Class2D quality is bad. You might consider changing the 'Minimal length (Å)' from 0 to a larger value (for example, 1000 Å) to improve the peaks in the histogram. If that does not help, you might consider redoing the Class2D task with longer extracted segments (>0.5x helical pitch) from longer filaments (> 1x pitch)."
        )

        @render.ui
        def _():
            if len(pair_distances()) > 0:
                download_ui = render.download(
                    label="Download selected helices", filename="helices.star"
                )

                @download_ui
                def download_retained_helices():
                    req(retained_helices_by_length())
                    indices = np.concatenate(
                        [h.index for hi, h in retained_helices_by_length()]
                    )
                    params_to_save = params().iloc[indices]

                    import starfile

                    d = dict(
                        optics=params_to_save.attrs["optics"], particles=params_to_save
                    )
                    yield starfile.to_string(d)

                return download_ui
            else:
                return None

    ui.HTML(
        "<i><p>Developed by the <a href='https://jiang.bio.purdue.edu/HelicalPitch' target='_blank'>Jiang Lab</a>. Report issues to <a href='https://github.com/jianglab/HelicalPitch/issues' target='_blank'>HelicalPitch@GitHub</a>.</p></i>"
    )


@reactive.effect
@reactive.event(input.run)
def get_class2d_from_upload():
    req(input.input_mode_classes() == "upload")
    fileinfo = input.upload_classes()
    class_file = fileinfo[0]["datapath"]
    try:
        data, apix = compute.get_class2d_from_file(class_file)
        nx = data.shape[-1]
    except Exception as e:
        print(e)
        data, apix = None, 0
        nx = 0
        m = ui.modal(
            f"failed to read the uploaded 2D class average images from {fileinfo[0]['name']}",
            title="File upload error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    data_all.set((data, apix))
    image_size.set(nx)


@reactive.effect
@reactive.event(input.run)
def get_class2d_from_url():
    req(input.input_mode_classes() == "url")
    req(len(input.url_classes()) > 0)
    url = input.url_classes()
    try:
        data, apix = compute.get_class2d_from_url(url)
        nx = data.shape[-1]
    except Exception as e:
        print(e)
        data, apix = None, 0
        nx = 0
        m = ui.modal(
            f"failed to download 2D class average images from {input.url_classes()}",
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    data_all.set((data, apix))
    image_size.set(nx)


@reactive.effect
@reactive.event(params, data_all, input.ignore_blank, input.sort_abundance)
def get_displayed_class_images():
    req(params() is not None)
    req(data_all() is not None)
    data, apix = data_all()
    n = len(data)
    images = [data[i] for i in range(n)]
    image_size.set(max(images[0].shape))

    try:
        df = params()
        abundance.set(compute.get_class_abundance(df, n))
    except Exception as e:
        print(e)
        m = ui.modal(
            f"Failed to get class abundance from the provided Class2D parameter and  image files. Make sure that the two files are for the same Class2D job",
            title="Information error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
        return None

    display_seq_all = np.arange(n, dtype=int)
    if input.sort_abundance():
        display_seq_all = np.argsort(abundance())[::-1]

    if input.ignore_blank():
        included = []
        for i in range(n):
            image = images[display_seq_all[i]]
            if np.max(image) > np.min(image):
                included.append(display_seq_all[i])
        images = [images[i] for i in included]
    else:
        included = display_seq_all
    image_labels = [f"{i+1}: {abundance()[i]:,d}" for i in included]

    displayed_class_ids.set(included)
    displayed_class_images.set(images)
    displayed_class_title.set(
        f"{len(included)}/{n} classes | {images[0].shape[1]}x{images[0].shape[0]} pixels | {apix} Å/pixel"
    )
    displayed_class_labels.set(image_labels)


@reactive.effect
@reactive.event(input.run)
def get_params_from_upload():
    req(input.input_mode_params() == "upload")
    fileinfo = input.upload_params()
    param_file = fileinfo[0]["datapath"]
    msg = None
    try:
        tmp_params = compute.get_class2d_params_from_file(param_file)
    except Exception as e:
        print(e)
        msg = str(e).replace(param_file, fileinfo[0]["name"])
        tmp_params = None
    params.set(tmp_params)

    if params() is None:
        if msg is None:
            msg = f"failed to parse the upload class2D parameters from {fileinfo[0]['name']}"
        msg = ui.markdown(
            msg.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br><br>")
        )
        m = ui.modal(
            msg,
            title="File upload error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)


@reactive.effect
@reactive.event(input.run)
def get_params_from_url():
    req(input.input_mode_params() == "url")
    url = input.url_params()
    msg = None
    try:
        tmp_params = compute.get_class2d_params_from_url(url)
    except Exception as e:
        print(e)
        msg = str(e)
        tmp_params = None
    params.set(tmp_params)

    if params() is None:
        if msg is None:
            msg = f"failed to download class2D parameters from {input.url_params()}"
        msg = ui.markdown(
            msg.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br><br>")
        )
        m = ui.modal(
            msg,
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)


selected_helices_min_len = reactive.value(([[], [], 0], 0))


@reactive.effect
@reactive.event(input.select_classes, params)
def get_selected_helices():
    req(params() is not None)
    req(image_size())
    req(len(abundance()))
    class_indices = [displayed_class_ids()[i] for i in input.select_classes()]
    helices = compute.select_classes(params=params(), class_indices=class_indices)
    if len(helices):
        class_indices2 = (
            np.unique(
                np.concatenate([h["rlnClassNumber"] for hi, h in helices])
            ).astype(int)
            - 1
        )
        assert set(class_indices) == set(class_indices2)

    if len(helices):
        filement_lengths = compute.get_filament_length(helices=helices)
        segments_count = np.sum([abundance()[i] for i in class_indices])
    else:
        filement_lengths = []
        segments_count = 0

    selected_helices.set((helices, filement_lengths, segments_count))
    if not input.auto_min_len():
        selected_helices_min_len.set((selected_helices(), input.min_len()))


@reactive.effect
@reactive.event(selected_helices)
def auto_set_filament_min_len():
    req(input.auto_min_len() is True)
    helices, filament_lengths, segments_count = selected_helices()
    _, min_len_tmp = compute.compute_pair_distances(
        helices=helices, lengths=filament_lengths, target_total_count=1000
    )
    min_len_tmp = int(min_len_tmp)
    ui.update_numeric("min_len", value=min_len_tmp)
    selected_helices_min_len.set((selected_helices(), min_len_tmp))


@reactive.effect
@reactive.event(input.min_len_changed)
def update_selected_helices_min_len():
    selected_helices_min_len.set((selected_helices(), input.min_len()))


@reactive.effect
@reactive.event(selected_helices_min_len, input.max_len)
def select_helices_by_length():
    previous = getattr(select_helices_by_length, "previous", ([], 0))
    selected_image_indices_previous, min_len_previous = previous
    (helices, filement_lengths, _), min_len = selected_helices_min_len()
    req(
        set(selected_image_indices_previous) != set(input.select_classes())
        or min_len_previous != min_len
    )
    if len(helices) == 0:
        retained_helices_by_length.set([])
    elif min_len == 0 and input.max_len() <= 0:
        retained_helices_by_length.set(helices)
    else:
        helices_retained, n_ptcls = compute.select_helices_by_length(
            helices=helices,
            lengths=filement_lengths,
            min_len=min_len,
            max_len=input.max_len(),
        )
        retained_helices_by_length.set(helices_retained)
    select_helices_by_length.previous = (input.select_classes(), min_len)


@reactive.effect
@reactive.event(retained_helices_by_length)
def get_pair_lengths():
    if len(retained_helices_by_length()):
        dists, _ = compute.compute_pair_distances(helices=retained_helices_by_length())
        pair_distances.set(dists)
    else:
        pair_distances.set([])


float_vars = dict(
    max_len=-1,
    max_pair_dist=-1,
    min_len=0,
    rise=4.75,
)
int_vars = dict(
    auto_min_len=1, bins=100, ignore_blank=1, show_sharable_url=0, sort_abundance=1
)
str_vars = dict(
    input_mode_classes="url",
    input_mode_params="url",
    url_params=urls[url_key][0],
    url_classes=urls[url_key][1],
)
all_input_vars = list(float_vars.keys()) + list(int_vars.keys()) + list(str_vars.keys())
reactive_vars_in = dict(select=(initial_selected_image_indices, int))
reactive_vars_out = dict(selected_image_indices=(input.select_classes, [0], "select"))

connection_made = reactive.Value(False)


@reactive.effect
@reactive.event(lambda: not connection_made())
def apply_initial_params_from_browser_url():
    d = helicon.shiny.get_client_url_query_params(input=input, keep_list=True)
    for k, v in d.items():
        if k in float_vars:
            v = list(map(float, v))
            if v[0] != float_vars[k]:
                if k in input:
                    ui.update_numeric(k, value=v[0])
        elif k in int_vars:
            v = list(map(int, v))
            if v[0] != int_vars[k]:
                if k in input:
                    ui.update_numeric(k, value=v[0])
        elif k in str_vars:
            if k in input:
                ui.update_text(k, value=v[0])
        elif k in reactive_vars_in:
            var, val_type = reactive_vars_in[k]
            v = list(map(val_type, v))
            var.set(v)
    if input.input_mode_params() == "url" and input.input_mode_classes() == "url":
        script = ui.tags.script(f"""document.getElementById('run').click();""")
        ui.insert_ui(ui=script, selector="body", where="afterEnd")


@render.ui
@reactive.event(
    *([input[k] for k in all_input_vars] + [v[0] for v in reactive_vars_out.values()])
)
def update_browser_url():
    if input.show_sharable_url():
        d = {}
        d.update(
            {
                k: float(input[k]())
                for k in float_vars
                if float_vars[k] != float(input[k]())
            }
        )
        d.update(
            {k: int(input[k]()) for k in int_vars if int_vars[k] != int(input[k]())}
        )
        d.update({k: input[k]() for k in str_vars if str_vars[k] != input[k]()})
        d.update(
            {
                var_url: var()
                for k, (var, val, var_url) in reactive_vars_out.items()
                if val != var()
            }
        )
        d = {k: d[k] for k in sorted(d.keys())}
    else:
        d = {}
    script = helicon.shiny.set_client_url_query_params(query_params=d)
    return script
