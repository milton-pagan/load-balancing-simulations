from tkinter import *


class MainWindow(object):

    def __init__(self):
        self.root = Tk()
        self.root.title("Simulation")

        self.top_frame = Frame(self.root)
        self.top_frame.pack(side='top')

        self.middle_frame_first = Frame(self.top_frame)
        self.middle_frame_first.pack(side='bottom')

        self.middle_frame_second = Frame(self.middle_frame_first)
        self.middle_frame_second.pack(side='bottom')

        self.bottom_frame = Frame(self.middle_frame_second)
        self.bottom_frame.pack(side='bottom')

        self.select = ''

    def add_widgets(self):
        # Adds all widgets to the window and positions them.
        title = Label(self.top_frame, text='Load Balancing Simulations', font='none 12 bold')
        title.pack()

        # Labels
        self.n_label = Label(self.middle_frame_first, text='Number of queues                   ')
        self.n_label.pack(side='left')
        self.lambda_label = Label(self.middle_frame_first, text='Lambda                 ')
        self.lambda_label.pack(side='left')
        self.arrivals_label = Label(self.middle_frame_first, text='Number of arrivals')
        self.arrivals_label.pack(side='left')

        # Entry boxes
        self.n_entry = Entry(self.middle_frame_second)
        self.n_entry.pack(side='left')

        self.lambda_entry = Entry(self.middle_frame_second)
        self.lambda_entry.pack(side='left')

        self.arrivals_entry = Entry(self.middle_frame_second)
        self.arrivals_entry.pack(side='left')

        # Buttons
        self.ht_button = Button(self.bottom_frame, text='Heavy Traffic', width=13, command=self.ht_clicked)
        self.ht_button.pack(side='left')

        self.mf_button = Button(self.bottom_frame, text='Mean Field', width=10, command=self.mf_clicked)
        self.mf_button.pack(side='left')

        self.hw_button = Button(self.bottom_frame, text='Halfin Witt', width=11, command=self.hw_clicked)
        self.hw_button.pack(side='left')

    def ht_clicked(self):
        self.select = 'ht'
        self.n = self.n_entry.get()
        self.lambda_ = self.lambda_entry.get()
        self.arrivals = self.arrivals_entry.get()

        if not self.is_number(self.n) or not self.is_number(self.lambda_) or not self.is_number(self.arrivals):
            sys.exit(ValueError)

        self.root.quit()

    def mf_clicked(self):
        self.select = 'mf'
        self.n = self.n_entry.get()
        self.lambda_ = self.lambda_entry.get()
        self.arrivals = self.arrivals_entry.get()

        if not self.is_number(self.n) or not self.is_number(self.lambda_) or not self.is_number(self.arrivals):
            sys.exit(ValueError)

        self.root.quit()

    def hw_clicked(self):
        self.select = 'hw'
        self.n = self.n_entry.get()
        self.lambda_ = self.lambda_entry.get()
        self.arrivals = self.arrivals_entry.get()

        if not self.is_number(self.n) or not self.is_number(self.lambda_) or not self.is_number(self.arrivals):
            sys.exit(ValueError)

        self.root.destroy()

    def run(self):
        self.add_widgets()
        self.root.mainloop()

        return self.select, int(self.n), float(self.lambda_), int(self.arrivals)

    @staticmethod
    def is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
