define([
    'base/js/namespace',
    'base/js/events',
    'jquery'
], function (Jupyter, events) {

    // Run on start
    function load_ipython_extension() {

        var all_events = [
            'app_initialized.DashboardApp',
            'app_initialized.NotebookApp',
            'autosave_disabled.Notebook',
            'autosave_enabled.Notebook',
            'before_save.Notebook',
            'changed',
            'checkpoint_created.Notebook',
            'checkpoint_delete_failed.Notebook',
            'checkpoint_deleted.Notebook',
            'checkpoint_failed.Notebook',
            'checkpoint_restore_failed.Notebook',
            'checkpoint_restored.Notebook',
            'checkpoints_listed.Notebook',
            'collapse_pager',
            'command_mode.Cell',
            'command_mode.Notebook',
            'config_changed.Editor',
            'create.Cell',
            'delete.Cell',
            'draw_notebook_list.NotebookList',
            'edit_mode.Cell',
            'edit_mode.Notebook',
            'execute.CodeCell',
            'execution_request.Kernel',
            'expand_pager',
            'file_load_failed.Editor',
            'file_loaded.Editor',
            'file_renamed.Editor',
            'file_saved.Editor',
            'file_saving.Editor',
            'input_reply.Kernel',
            'kernel_autorestarting.Kernel',
            'kernel_busy.Kernel',
            'kernel_connected.Kernel',
            'kernel_connection_dead.Kernel',
            'kernel_connection_failed.Kernel',
            'kernel_created.Kernel',
            'kernel_created.Session',
            'kernel_dead.Kernel',
            'kernel_dead.Session',
            'kernel_disconnected.Kernel',
            'kernel_idle.Kernel',
            'kernel_interrupting.Kernel',
            'kernel_killed.Kernel',
            'kernel_killed.Session',
            'kernel_ready.Kernel',
            'kernel_reconnecting.Kernel',
            'kernel_restarting.Kernel',
            'kernel_starting.Kernel',
            'kernelspecs_loaded.KernelSpec',
            'list_checkpoints_failed.Notebook',
            'mode_changed.Editor',
            'no_kernel.Kernel',
            'notebook_copy_failed',
            'notebook_deleted.NotebookList',
            'notebook_load_failed.Notebook',
            'notebook_loaded.Notebook',
            'notebook_loading.Notebook',
            'notebook_read_only.Notebook',
            'notebook_renamed.Notebook',
            'notebook_restoring.Notebook',
            'notebook_save_failed.Notebook',
            'notebook_saved.Notebook',
            'open_with_text.Pager',
            'output_appended.OutputArea',
            'preset_activated.CellToolbar',
            'preset_added.CellToolbar',
            'rebuild.QuickHelp',
            'received_unsolicited_message.Kernel',
            'rendered.MarkdownCell',
            'resize',
            'resize-header.Page',
            'save_status_clean.Editor',
            'save_status_dirty.Editor',
            'select.Cell',
            'selected_cell_type_changed.Notebook',
            'send_input_reply.Kernel',
            'sessions_loaded.Dashboard',
            'set_dirty.Notebook',
            'set_next_input.Notebook',
            'shell_reply.Kernel',
            'spec_changed.Kernel',
            'spec_match_found.Kernel',
            'spec_not_found.Kernel',
            'trust_changed.Notebook',
            'unrecognized_cell.Cell',
            'unrecognized_output.OutputArea',
            'unregistered_preset.CellToolbar',
        ];

        Jupyter.notebook.events.on(all_events.join(' '), function(evt, data) {
            var seen = [];

            var jsonData = JSON.stringify(data, function(key, val) {
               if (val != null && typeof val == "object") {
                    if (seen.indexOf(val) >= 0) {
                        return;
                    }
                    seen.push(val);
                }
                return val;
            });

            var log = {
                "evt": evt.type,
                "datetime": (new Date()).toISOString(),
                "filename": (Jupyter.notebook.notebook_path).split('.ipynb')[0],
                "data": jsonData
            }
            console.log(JSON.stringify(log));
            var baseurl = window.location.origin

            var xhr = new XMLHttpRequest();
            xhr.open("POST", baseurl + '/savelogs', true);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.send(JSON.stringify({
                value: log
            }));

        });
    }
    return {
        load_ipython_extension: load_ipython_extension
    };
});