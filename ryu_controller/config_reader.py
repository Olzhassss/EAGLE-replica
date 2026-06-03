import configparser

class Reader:

    def __init__(self, filename):
        self.config = configparser.ConfigParser()
        self.config.read(filename)

    def first_switch(self):
        return self.config.getint('DEFAULT', 'switch_1')

    def second_switch(self):
        return self.config.getint('DEFAULT', 'switch_2')

    def switches_to_monitor(self):
        list_sw = self.config.get('DEFAULT', 'switches_monitored').split(',')
        list_sw = list(map(int, list_sw))
        return list_sw

    def macs_to_monitor(self):
        list_macs = self.config.get('DEFAULT', 'macs_interested').split(',')
        return list_macs

    def frequency_update(self):
        return self.config.getint('DEFAULT', 'frequency_update')

    def frequency_request(self):
        return self.config.getint('DEFAULT', 'frequency_data')

    def percentage_network(self):
        return self.config.getint('DEFAULT', 'percentage_of_known_network')

    def in_port(self, switch):
        return self.config.getint('OVS_{}'.format(switch), 'in_port')

    def out_port(self, switch):
        return self.config.getint('OVS_{}'.format(switch), 'out_port')
