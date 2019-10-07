#!/usr/bin/python
# -*- coding: utf-8 -*-
# This is the CS 352 Spring 2017 Client for the 1st programming
# project
 

import argparse
import time
import struct 
import md5
import os 
import sock352
import sys            
import binascii
import socket as syssock        
import random
from thread import*


#Ningjie Zhang Start Code
#The bit flags needed are set in the flags field of the packet header. The exact bit definitions of the flags are: 
SOCK352_SYN = 0x01
SOCK352_FIN = 0x02
SOCK352_ACK = 0x04
SOCK352_RESET = 0x08
SOCK352_HAS_OPT =0xA0

VERSION = 0x1
flags = SOCK352_SYN
opt_ptr = 0
protocol = 0
header_len = 0
checksum = 0
source_port = 0
dest_port = 0
sequence_no = random.randint(0, 1000)
ack_no = sequence_no + 1
window = 0
payload_len = 0


global udpsendport            #set three global variables so that all
global udprecvport            #the function in this program could get
global udpsock                #access to it.

def init(udp_port1, udp_port2):
    global udpsendport
    global udprecvport
    global udpsock

    udpsendport = int(udp_port1)  #set global var to the input
    udprecvport = int(udp_port2)
    udpsock = syssock.socket(syssock.AF_INET, syssock.SOCK_DGRAM)#Create a new socket using the given address family, socket type and protocol number.
    udpsock.settimeout(0.2)
    
    if udpsendport == 0:
        udpsendport = 27181
        
    if udprecvport == 0:
        udprecvport = 27182
    
        print('socket has been initialized')
    return    
    
class socket:
    def __init__(self):  # fill in your code here
        global udpsendport
        global udprecvport
        global udpsock
    
        self.send_port = udpsendport
        self.recv_port = udprecvport
        self.sock = udpsock
        
        self.sock352PktHdrData = '!BBBBHHLLQQLL'
        self.udpPkt_hdr_data = struct.Struct(self.sock352PktHdrData)

        print('socket class initialized')
        return
        
    def bind(self, address): #可以用来将 socket 绑定到特定的地址和端口上
        #print('<in bind>')
        #self.sock.bind((address[0],self.recv_port))
        return
    
    def connect(self, address):
       
        self.sock.bind(('',self.recv_port))
    	  
        seq_no = random.randint(0,100)
        print 'connect'
        syn_header = self.udpPkt_hdr_data.pack(VERSION, flags, opt_ptr, protocol, header_len, checksum, source_port, dest_port, seq_no, ack_no, window, payload_len)
        while True:
            self.sock.sendto(syn_header,(address[0],self.send_port))
            print'message was sent'
            try:
                syn_ack_data, addr = self.sock.recvfrom(2048)
                #a, flags, c, d, e, f, g, h, i, ack, k, l = struct.unpack(self.sock352PktHdrData, syn_ack_data)
                #if flags & 0x1 == 1 and ack == 1:
                 #   print 'connected'
                  #  return
                #break            
            except syssock.timeout:
                print('timeout error')
                continue
            syn_ack = self.pkt_hdr.unpack(syn_ack_data)
            if syn_ack[FLAGS] == (SOCK352_ACK | SOCK352_SYN) and syn_ack[ACK_NO] == seq_no+1:
               break         
        #self.socket.connect(address) 
        ack = self.udpPkt_hdr_data.pack(VERSION, SOCK352_ACK, 0, 0, 2048, 0, 0, 0, seq_no+1, syn_ack[sequence_no]+1, 0, 0)
                
        self.sock.sendto(ack, (address[0], self.send_port))
        self.conn = (address[0], self.send_port)
        return 
    
    
    def listen(self,backlog): #listen 可以将 socket 置于监听模式：
        return 

    def accept(self): #当有客户端向服务器发送连接请求时，服务器会接收连接：
        print('in Accept()')
        while True:
            try:
                #print('in try')
                syn_header_data, addr = self.sock.recvfrom(2048)
                print (syn_header_data)
            except:
                print('continue')
                continue
            syn_header = self.udpPkt_hdr_data.unpack(syn_header_data)
            if syn_header[flages] == SOCK352_SYN:
                break
        syn_ack = self.udpPkt_hdr_data.pack(VERSION,SOCK352_SYN | SOCK352_ACK,0,0,40,0,0,0, random.randint(0,100), syn_header[sequence_no]+1,0,0)
        self.sock.sendto(syn_ack, addr)
        try:
            ack, addr = self.sock.recvfrom(2048) 
        except syssock.timeout:
            pass
               
        self.conn = (addr[0], self.send_port)
        (clientsocket, address) = (self.sock, addr)	
        return (clientsocket,address)

            
        

    def close(self):
        sock.socket.shutdown(socket.SHUT_RDWR)
        sock.socket.close(self)
        return

    def send(self, buffer):
        bytessent = 0  # fill in your code here
        print 'here in send'
        bytesAcp = bytessent
        while bytessent < buffer:
            sent = self.sock.send(buffer[bytesAcp])
            if sent == 0:
                raise RuntimeError("send failed")
            bytessent = bytessent + sent
            bytesAcp = bytessent
        return bytesAcp

    def recv(self, nbytes): 
        nbytetwo = []
        print 'in recv'
        bytesreceived = 0
        while bytesreceived < MSGLEN:
            nbytetwo = self.sock.recv(min(MSGLEN - bytesreceived, 2048))
            if nbytetwo == '':
                raise RuntimeError("socket connection broke down");
            nbytetwo.append(nbytetwo)
            bytesreceived = bytesreceived + len(nbytetwo)
        return bytesreceived



#Ningjie Zhang End Code





def main():
    # parse all the arguments to the client 
    parser = argparse.ArgumentParser(description='CS 352 Socket Client')
    parser.add_argument('-f','--filename', help='File to Send', required=False)
    parser.add_argument('-d','--destination', help='Destination IP Host', required=True)
    parser.add_argument('-p','--port', help='remote sock352 port', required=False)
    parser.add_argument('-u','--udpportRx', help='UDP port to use for receiving', required=True)
    parser.add_argument('-v','--udpportTx', help='UDP port to use for sending', required=False)

    # get the arguments into local variables 
    args = vars(parser.parse_args())
    filename = args['filename']
    destination = args['destination']
    udpportRx = args['udpportRx']

    if (args['udpportTx']):
        udpportTx = args['udpportTx']
    else:
        udpportTx = ''
        
    # the port is not used in part 1 assignment, except as a placeholder
    if (args['port']): 
        port = args['port']
    else:
        port = 1111 

    # open the file for reading
    if (filename):
        try: 
            filesize = os.path.getsize(filename)
            fd = open(filename, "rb")
            usefile = True
        except:
            print ( "error opening file: %s" % (filename))
            exit(-1)
    else:
        pass 

    # This is where we set the transmit and receive
    # ports the client uses for the underlying UDP
    # sockets. If we are running the client and
    # server on the same machine, these ports
    # need to be different. If they are running on
    # different machines, we can re-use the same
    # ports. 
    if (udpportTx):
        sock352.init(udpportTx,udpportRx)
    else:
        sock352.init(udpportRx,udpportRx)

    # create a socket and connect to the remote server
    s = sock352.socket()
    s.connect((destination,port))
    
    # send the size of the file as a 4 byte integer
    # to the server, so it knows how much to read
    FRAGMENTSIZE = 8192
    longPacker = struct.Struct("!L")
    fileLenPacked = longPacker.pack(filesize);
    s.send(fileLenPacked)

    # use the MD5 hash algorithm to validate all the data is correct
    mdhash = md5.new()

    # loop for the size of the file, sending the fragments 
    bytes_to_send = filesize

    start_stamp = time.clock()    
    while (bytes_to_send > 0):
        fragment = fd.read(FRAGMENTSIZE)
        mdhash.update(fragment)
        totalsent = 0
        # make sure we sent the whole fragment 
        while (totalsent < len(fragment)):
            sent = s.send(fragment[totalsent:])
            if (sent == 0):
                raise RuntimeError("socket broken")
            totalsent = totalsent + sent
        bytes_to_send = bytes_to_send - len(fragment)

    end_stamp = time.clock() 
    lapsed_seconds = end_stamp - start_stamp
    
    # this part send the lenght of the digest, then the
    # digest. It will be check on the server 
    
    digest = mdhash.digest()
    # send the length of the digest
    long = len(digest)
    digestLenPacked = longPacker.pack(long)
    sent = s.send(digestLenPacked)
    if (sent != 4):
        raise RuntimeError("socket broken")
    
    # send the digest 
    sent = s.send(digest)
    if (sent != len(digest)):
        raise RuntimeError("socket broken")

    if (lapsed_seconds > 0.0):
        print ("client1: sent %d bytes in %0.6f seconds, %0.6f MB/s " % (filesize, lapsed_seconds,
(filesize/lapsed_seconds)/(1024*1024)))
    else:
        print ("client1: sent %d bytes in %d seconds, inf MB/s " % (filesize, lapsed_seconds))        

    fd.close()
    s.close()
# this gives a main function in Python
if __name__ == "__main__":
    main()
