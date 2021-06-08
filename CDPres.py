import time
import os
import wx
import numpy as np
import unidec

from GUniDec import UniDecApp
from pubsub import pub
import unidec_modules.unidectools as ud
import unidec_modules.CDEng as CDEng
from unidec_modules.gui_elements import CDWindow
import multiprocessing
from unidec_modules.unidec_presbase import UniDecPres
from unidec_modules import peakwidthtools, CDCal
import platform


class UniDecCDApp(UniDecApp):
    """
    Main UniDec GUI Application.
    Presenter contains UniDec engine at self.eng and main GUI window at self.view
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize App
        :param args:
        :param kwargs:
        :return: UniDecApp object
        """
        UniDecPres.__init__(self, *args, **kwargs)
        self.init(*args, **kwargs)

    def init(self, *args, **kwargs):
        """
        Initialize Engine and View. Load defaults.
        :param args:
        :param kwargs:
        :return:
        """
        self.eng = CDEng.UniDecCD()
        self.view = CDWindow.CDMainwindow(self, "UCD: UniDec for Charge Detection-Mass Spectrometry", self.eng.config)
        self.comparedata = None

        pub.subscribe(self.on_get_mzlimits, 'mzlimits')
        '''
        pub.subscribe(self.on_integrate, 'integrate')
        pub.subscribe(self.on_left_click, 'left_click')'''
        self.eng.config.recentfile = self.eng.config.recentfileCD
        self.recent_files = self.read_recent()
        self.cleanup_recent_file(self.recent_files)
        self.view.menu.update_recent()


        self.on_load_default(0)

        if "path" in kwargs:
            newdir, fname = os.path.split(kwargs["path"])
            self.on_open_file(fname, newdir)
            # self.on_dataprep_button(0)
            # self.on_auto(0)

        if self.infile is not None:
            newdir, fname = os.path.split(self.infile)
            self.on_open_file(fname, newdir)
            # self.on_dataprep_button(0)
            # self.on_auto(0)

        if False and platform.node() == 'DESKTOP-08TGCJO':
            path = "C:\\Data\\CDMS\\02242021_MK_BSA__CD-MS_Aqu_ACN_10min.RAW"
            path = "C:\\Data\\CDMS\\20210309_MK_ADH_pos_CDMS_512ms_5min_50ms_pressure01.RAW"
            # path = "C:\\Data\\CDMS\\Replicates\\AAV8_IMID_CDMS_1.RAW"
            # path = 'C:\\Data\\CDMS\\CDMSJarrold\\data_Dec1_x8.txt'
            # path = "C:\\Data\\CDMS\\Replicates\\DPPC_CDMS_1.RAW"
            # path = "C:\Data\Wendy\FW CDMS runs20210322081451\\20210301_OBJ41415_CDMS_Pure_4.mzML.gz"
            self.on_open_file(None, None, path=path)

    def on_open(self, e=None):
        """
        Open dialog for file opening
        :param e: unused space for event
        :return: None
        """
        dlg = wx.FileDialog(self.view, "Choose a data file in x y list, mzML, or Thermo Raw format", '', "", "*.*")
        if dlg.ShowModal() == wx.ID_OK:
            self.view.SetStatusText("Opening", number=5)
            filename = dlg.GetFilename()
            print("Opening: ", filename)
            if os.path.splitext(filename)[1] == ".zip":
                print("Can't open zip, try Load State.")
                return
            dirname = dlg.GetDirectory()
            self.on_open_file(filename, dirname)
        dlg.Destroy()

    def on_open_file(self, filename, directory, path=None):
        """
        Opens a file. Run self.eng.open_file.
        :param filename: File name
        :param directory: Directory containing file
        :param skipengine: Boolean, Whether to skip running the engine (used when loading state)
        :return: None
        """
        # tstart =time.perf_counter()

        # Clear other plots and panels
        self.view.peakpanel.clear_list()
        self.view.clear_all_plots()

        if path is None:
            path = os.path.join(directory, filename)
        print("Opening", path)
        self.eng.open_file(path)

        # Set Status Bar Text Values
        self.view.SetStatusText("File: " + path, number=1)
        self.view.SetStatusText("Data Length: " + str(len(self.eng.farray)), number=2)
        # self.view.SetStatusText("R\u00B2 ", number=3)
        # Update view with data limits
        if self.eng.config.batchflag != 1:
            self.view.controls.ctlminmz.SetValue(str(np.amin(self.eng.config.minmz)))
            self.view.controls.ctlmaxmz.SetValue(str(np.amax(self.eng.config.maxmz)))

        # Load Config to GUI
        self.import_config()

        self.view.SetStatusText("Ready", number=5)

        self.write_to_recent()
        self.view.menu.update_recent()

        self.on_dataprep_button()

    def on_dataprep_button(self, e=None):
        self.view.clear_all_plots()
        self.export_config(self.eng.config.confname)
        self.eng.process_data()
        if len(self.eng.harray) > 0:
            self.makeplot1()
            self.makeplot2()
            self.makeplot3()
            self.makeplot4()
            self.view.SetStatusText("Data Length: " + str(len(self.eng.farray)), number=2)

    def makeplot1(self, e=None):
        if not ud.isempty(self.eng.harray) or np.amax(self.eng.harray) == 0:
            self.view.plot1.contourplot(xvals=self.eng.mz, yvals=self.eng.ztab, zgrid=np.transpose(self.eng.harray),
                                        config=self.eng.config)
        else:
            print("ERROR: Histogram Array is empty")

    def plotkernel(self, e=None):
        self.export_config()
        self.eng.make_kernel()
        kernel = np.transpose(self.eng.kernel)
        kernel = np.roll(kernel, (int(kernel.shape[0]/2), int(kernel.shape[1]/2)), axis=(0, 1))
        if not ud.isempty(kernel) or np.amax(kernel) == 0:
            self.view.plot1.contourplot(xvals=self.eng.mz, yvals=self.eng.ztab, zgrid=kernel,
                                        config=self.eng.config)
        else:
            print("ERROR: Kernel Array is empty")

    def plotnoise(self, e=None):
        print("Plotting Noise")
        x = self.eng.mz
        y = np.ones_like(x) * self.eng.noise
        y = self.eng.simp_convert(y)

        self.view.plot1.subplot1.plot(x, y, color="r")
        self.view.plot1.repaint()

    def makeplot3(self, e=None):
        self.view.plot3.plotrefreshtop(self.eng.data.zdat[:, 0], self.eng.data.zdat[:, 1], config=self.eng.config,
                                       xlabel="Charge")
        pass

    def on_unidec_button(self, e=None):
        self.view.SetStatusText("Deconvolving", number=5)
        self.view.clear_all_plots()
        self.export_config(self.eng.config.confname)
        self.eng.run_deconvolution()
        self.makeplot1()
        self.makeplot2()
        self.makeplot3()
        self.makeplot4()
        self.view.SetStatusText("Finished", number=5)
        pass

    def on_pick_peaks(self, e=None):
        """
                Pick peaks and perform initial plots on them.
                :param e: unused space for event
                :return: None
                """
        print("Peak Picking")
        self.view.SetStatusText("Detecting Peaks", number=5)
        tstart = time.perf_counter()
        self.export_config(self.eng.config.confname)
        self.eng.pick_peaks()
        self.view.SetStatusText("Plotting Peaks", number=5)
        if self.eng.config.batchflag == 0:
            self.view.peakpanel.add_data(self.eng.pks)
            self.makeplot2(1)
            self.makeplot6(1)
            self.makeplot4(1)
        self.view.SetStatusText("Peak Pick Done", number=5)
        self.on_score()
        pass

    def on_replot(self, e=None):
        self.makeplot1()
        self.makeplot2()
        self.makeplot3()
        self.makeplot4()
        self.makeplot6()
        pass

    def on_peak_width_tool(self, e=None):
        """
        Open peak width tool window. After it has returned, update the GUI to reflect the new peak widths.
        :param e: unused event
        :return: None
        """
        self.export_config()
        if not ud.isempty(self.eng.data.data2):
            self.export_config(None)
            dlg = peakwidthtools.PeakTools2d(self.view)
            dlg.initialize_interface(self.eng.data.mzgrid, self.eng.data.data2, self.eng.config)
            dlg.ShowModal()
            self.eng.config.csig = self.eng.config.dtsig
            self.import_config(None)
        else:
            print("Need to process data first")
        pass

    def on_compare(self, e=None):
        """
        Open dialog for file opening
        :param e: unused space for event
        :return: None
        """
        dlg = wx.FileDialog(self.view, "Choose a data file in x y list, mzML, or Thermo Raw format", '', "", "*.*")
        if dlg.ShowModal() == wx.ID_OK:
            self.view.SetStatusText("Opening", number=5)
            filename = dlg.GetFilename()
            print("Opening: ", filename)
            if os.path.splitext(filename)[1] == ".zip":
                print("Can't open zip, try Load State.")
                dlg.Destroy()
                return
            dirname = dlg.GetDirectory()
        else:
            dlg.Destroy()
            return
        dlg.Destroy()

        udeng = unidec.UniDec()
        udeng.open_file(filename, dirname)
        udeng.unidec_imports()
        self.comparedata = udeng.data.massdat

        self.on_compare2()

    def on_compare2(self, e=None):
        if not ud.isempty(self.comparedata):
            norm = np.amax(self.eng.data.massdat[:, 1]) / np.amax(self.comparedata[:, 1])

            self.view.plot2.plotadd(self.comparedata[:, 0], self.comparedata[:, 1] * norm, colval="g", nopaint=False)
        else:
            self.on_compare()

    def on_compare_proc(self, e=None):
        self.comparedata = self.eng.unprocessed
        self.on_compare2()

    def on_calibrate(self, e=None):
        dlg = CDCal.CDCalDialog(self.view)
        dlg.initialize_interface(self.eng.config)

    def on_gpu_mode(self, gpumode=False):
        self.eng.gpu_mode(gpumode)

    def remake_mainwindow(self, tabbed=None):
        iconfile = self.view.icon_path
        # evt=EventManager()
        # print evt.GetStats()
        wx.GetApp().Yield()
        self.view.on_exit()
        self.view = []
        self.view = CDWindow.CDMainwindow(self, "UCD: UniDec for Charge Detection-Mass Spectrometry", self.eng.config,
                                          iconfile=iconfile, tabbed=tabbed)
        self.view.Show()
        self.view.import_config_to_gui()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = UniDecCDApp()
    app.start()