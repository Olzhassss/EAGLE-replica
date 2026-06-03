#!/usr/bin/python
from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import TCLink
net=None

def topology():

    net = Mininet(controller=Controller, link=TCLink, switch=OVSKernelSwitch)
    c0 = net.addController('c0', controller=RemoteController, port=6633)
    c1 = net.addController('c1', controller=RemoteController, port=6634)
    c2 = net.addController('c2', controller=RemoteController, port=6635)

    s1 = net.addSwitch('s1')
    s2 = net.addSwitch('s2')
    s3 = net.addSwitch('s3')
    s4 = net.addSwitch('s4')
    s5 = net.addSwitch('s5')
    s6 = net.addSwitch('s6')
    s7 = net.addSwitch('s7')
    s8 = net.addSwitch('s8')
    s9 = net.addSwitch('s9')
    s10 = net.addSwitch('s10')

    # Few spare switches for assisting during healing phase
    s11 = net.addSwitch('s11')
    s12 = net.addSwitch('s12')
    s13 = net.addSwitch('s13')

    h1 = net.addHost('h1', mac='f6:c0:28:8f:a5:9a')
    net.addLink(s1, h1, bw=100)
    h2 = net.addHost('h2', mac='d2:77:87:57:5d:69')
    net.addLink(s2, h2, bw=100)
    h3 = net.addHost('h3', mac='8a:ae:bc:42:55:d8')
    net.addLink(s3, h3, bw=100)
    h4 = net.addHost('h4', mac='f6:c3:e5:59:2a:6d')
    net.addLink(s4, h4, bw=100)
    h5 = net.addHost('h5', mac='5a:0d:09:aa:cd:f8')
    net.addLink(s5, h5, bw=100)
    h6 = net.addHost('h6', mac='96:f3:f8:b7:84:ab')
    net.addLink(s6, h6, bw=100)
    h7 = net.addHost('h7', mac='72:9e:d8:59:8b:d0')
    net.addLink(s7, h7, bw=100)
    h8 = net.addHost('h8', mac='1e:75:d8:41:51:f9')
    net.addLink(s5, h8, bw=100)
    h9 = net.addHost('h9', mac='4a:e2:74:fa:3e:c7')
    net.addLink(s5, h9, bw=100)

    net.addLink(s1, s6, bw=65)
    net.addLink(s2, s6, bw=64)
    net.addLink(s3, s7, bw=63)
    net.addLink(s4, s7, bw=65)
    net.addLink(s5, s8, bw=70)
    net.addLink(s6, s9, bw=58)
    net.addLink(s7, s10, bw=73)
    net.addLink(s8, s9, bw=97)
    net.addLink(s9, s10, bw=94)
    net.addLink(s10, s8, bw=99)
    
    net.addLink(s5, s11, bw=100)
    net.addLink(s11, s9, bw=100)
   
    net.addLink(s6, s12, bw=100)
    net.addLink(s12, s10, bw=100)
   
    net.addLink(s7, s13, bw=100)
    net.addLink(s13, s8, bw=100)

    net.start()
    set_arp_hosts(net)
    net.staticArp()
    c0.start()
    c1.start()
    c2.start()
    s1.start([c0])
    s2.start([c0])
    s3.start([c2])
    s4.start([c2])
    s5.start([c1])
    s6.start([c0])
    s7.start([c2])
    s8.start([c2])
    s9.start([c1])
    s10.start([c0])
    s11.start([c1])
    s12.start([c0])
    s13.start([c2])

    CLI(net)
    return net

def set_arp_hosts(net):
    for src in net.hosts:
        for dst in net.hosts:
            if src != dst:
                src.setARP(dst.IP(), dst.MAC())


if __name__ == '__main__':
    setLogLevel('info')
    net = topology()
    net.stop()
