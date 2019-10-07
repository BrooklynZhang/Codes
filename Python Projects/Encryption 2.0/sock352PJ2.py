
# CS 352 project part 2 
# this is the initial socket library for project 2 
# You wil need to fill in the various methods in this
# library 

# main libraries 
import binascii
import socket as syssock
import struct
import sys

#additional libraries
import random

# encryption libraries 
import nacl.utils
import nacl.secret
import nacl.utils
from nacl.public import PrivateKey, Box

# if you want to debug and print the current stack frame 
from inspect import currentframe, getframeinfo

# these are globals to the sock352 class and
# define the UDP ports all messages are sent
# and received from

# the ports to use for the sock352 messages 
global sock352portTx
global sock352portRx
# the public and private keychains in hex format 
global publicKeysHex
global privateKeysHex

# the public and private keychains in binary format 
global publicKeys
global privateKeys

# the encryption flag 
global ENCRYPT

publicKeysHex = {} 
privateKeysHex = {} 
publicKeys = {} 
privateKeys = {}

# this is 0xEC 
ENCRYPT = 236 

# this is the structure of the sock352 packet 
sock352HdrStructStr = '!BBBBHHLLQQLL'

SOCK352_SYN = 0x01
SOCK352_FIN = 0x02
SOCK352_ACK = 0x04
SOCK352_RESET = 0x08
SOCK352_HAS_OPT = 0xA0

MTU = 64000

version = 0x1
flags = SOCK352_SYN
#opt_ptr = 0
protocol = 0
header_len = struct.calcsize('!BBBBHHLLQQLL')
checksum = 0
source_port = 0
dest_port = 0
sequence_no = random.randint(1, sys.maxint)
ack_no = 0
window = 0
payload_len = 0



def init(UDPportTx,UDPportRx):
    global sock352portTx
    global sock352portRx

    # create the sockets to send and receive UDP packets on 
    # if the ports are not equal, create two sockets, one for Tx and one for Rx
    
    #start Coding
    if (UDPportTx ==0 ):
        UDPportTx = 27182
    if (UDPportRx ==0 ):
        UDPportRx = 27182
        
    sock352portTx = UDPportTx
    sock352protRx = UDPportRx
    
    return #end coding
    
# read the keyfile. The result should be a private key and a keychain of
# public keys
def readKeyChain(filename):
    global publicKeysHex
    global privateKeysHex 
    global publicKeys
    global privateKeys 
    
    if (filename):
        try:
            keyfile_fd = open(filename,"r")
            for line in keyfile_fd:
                words = line.split()
                # check if a comment
                # more than 2 words, and the first word does not have a
                # hash, we may have a valid host/key pair in the keychain
                if ( (len(words) >= 4) and (words[0].find("#") == -1)):
                    host = words[1]
                    #if host == "127.0.0.1":
                    #host = "localhost"
                    port = words[2]
                    keyInHex = words[3]
                    if (words[0] == "private"):
                        privateKeysHex[(host,port)] = keyInHex
                        privateKeys[(host,port)] = nacl.public.PrivateKey(keyInHex, nacl.encoding.HexEncoder)
                    elif (words[0] == "public"):
                        publicKeysHex[(host,port)] = keyInHex
                        publicKeys[(host,port)] = nacl.public.PublicKey(keyInHex, nacl.encoding.HexEncoder)
        except Exception,e:
            print ( "error: opening keychain file: %s %s" % (filename,repr(e)))
    else:
            print ("error: No filename presented")             

    return (publicKeys,privateKeys)

class socket:
    
    def __init__(self):
        # your code goes here 
        self.sock = syssock.socket(syssock.AF_INET, syssock.SOCK_DGRAM)
        self.sock.settimeout(0.2)
        self.client_addr = None
        self.serv_addr = None
        self.last_pkt_recvd = None
        self.CheckConnected = False
        self.udpPkt_hdr_data = struct.Struct(sock352HdrStructStr)
        self.CheckServer = None
        self.encrypt = False
        self.box = None
        self.nonce_flag = 0        
        
        return 
        
    def bind(self,address):
        # bind is not used in this assignment 
        return

    def connect(self,*args):
        self.CheckServer = False
        address = args[0]
        self.serv_addr = (address[0], int(Txport))
        self.sock.bind(('', int(Rxport)))        
        # example code to parse an argument list 
        global sock352portTx
        global ENCRYPT
        if (len(args) >= 1): 
            (host,port) = args[0]
        if (len(args) >= 2):
            if (args[1] == ENCRYPT):
                self.encrypt = True
                client_private_key = privateKeys[('*', '*')]
                server_public_key = publicKeys[(self.serv_addr[0], str(self.serv_addr[1]))]
                self.box = Box(client_private_key, server_public_key)
                opt_ptr = 0b1
        else:
            opt_ptr = 0b0 
                
        # your code goes here
        SYN_Packet = self.udpPkt_hdr_data.pack(version, flags, opt_ptr, protocol, header_len, checksum, source_port,
                                                       dest_port, sequence_no, ack_no, window, payload_len)        

        acked = False
        while not acked:
            try:
                self.sock.sendto(SYN_Packet, self.serv_addr)
                #print("Connection request sent")
                ack = self.sock.recvfrom(header_len)[0]
                acked = True
            except syssock.timeout:
                #print("Timeout occurred. Resending...")
                pass
        self.last_pkt_recvd = struct.unpack('!BBBBHHLLQQLL', ack)
            #print("Server response received")
        if self.last_pkt_recvd [1] == SOCK352_RESET:
                #print("Connection Refused")
            pass
        elif self.last_pkt_recvd [1] == SOCK352_SYN | SOCK352_ACK:
            #print("Connection Successful")
            pass
        else:
            #print("Server response invalid")
            pass
        return        

    def listen(self,backlog):
        # listen is not used in this assignments 
        pass
    

    def accept(self,*args):
        self.CheckServer = True
        self.sock.settimeout(None)
        # example code to parse an argument list 
        global ENCRYPT
        SYN_Packet, self.client_addr = self.sock.recvfrom(int(header_len))
        if (len(args) >= 1):
            if (args[0] == ENCRYPT):
                self.encrypt = True
                server_private_key = privateKeys[('*', '*')]
                client_public_key = publicKeys[(self.client_addr[0], str(self.client_addr[1]))]
                self.box = Box(server_private_key, client_public_key)
                opt_ptr = 0b1
        else:
            opt_ptr = 0b0
        self.last_pkt_recvd = struct.unpack('!BBBBHHLLQQLL', SYN_Packet)
        # your code goes here 
        if self.CheckConnected:
            flags = SOCK352_RESET
        else:
            flags = SOCK352_SYN | SOCK352_ACK
            self.CheckConnected = True        
        connection_response = self.udpPkt_hdr_data.pack(0x1, flags, opt_ptr, 0, struct.calcsize('!BBBBHHLLQQLL'), 0, 0 , 0, self.last_pkt_recvd[8] + 1, 0, 0)
        self.sock.sendto(connection_response, self.client_addr)
        
    def close(self):
        # your code goes here 
        version = 0x1
        flags = SOCK352_FIN
        opt_ptr = 0
        protocol = 0
        header_len = struct.calcsize('!BBBBHHLLQQLL')
        checksum = 0
        source_port = 0
        dest_port = 0
        sequence_no = self.last_pkt_recvd[8] + 1
        ack_no = 0
        window = 0
        payload_len = 0
        connection_response = self.udpPkt_hdr_data.pack(version, flags, opt_ptr, protocol, header_len, checksum,
                                                        source_port, dest_port, sequence_no, ack_no, window,
                                                        payload_len)
        acked = False
        while not acked:
            try:
                if self.CheckServer == True:
                    self.sock.sendto(connection_response, self.client_addr)
                elif self.CheckServer == False:
                    self.sock.sendto(connection_response, self.serv_addr)
                            #print("Termination request sent")
                    ack = self.sock.recvfrom(header_len)[0]
                    acked = True
            except syssock.timeout:
                pass              
            FIN_ACK = struct.unpack('!BBBBHHLLQQLL', ack)
            if FIN_ACK[1] == SOCK352_FIN:
                flags |= SOCK352_ACK
                self.sock.close()
                #print("Connection terminated")
            else:
                #print("Server response invalid")
                pass       
        return 

    def send(self,buffer):
        # your code goes here 
        plaintext_len = len(buffer)
        
        if len(buffer) == 0:
            return 0
        
        buffer = buffer[:4000]  # added to solve fragmentation problem
        
        if self.encrypt == True:
            if self.nonce_flag == 0:
                nonce = nacl.utils.random(Box.NONCE_SIZE)
                buffer = self.box.encrypt(buffer, nonce)
                self.nonce_flag = 1
            else:
                buffer = self.box.encrypt(buffer)
                opt_ptr = 0b1
        else:
            opt_ptr = 0b0        
        version = 0x1
        flags = 0
        #opt_ptr = 0
        protocol = 0
        header_len = struct.calcsize('!BBBBHHLLQQLL')
        checksum = 0
        source_port = 0
        dest_port = 0
        sequence_no = self.last_pkt_recvd[8] + 1
        ack_no = 0
        window = 0
        payload_len = len(buffer)

        bytes_to_send = len(buffer)
        bytessent = 0
        if len(buffer) <= MTU:
            # packs header data into a string suitable to be sent over transmitting socket
            header = self.udpPkt_hdr_data.pack(version, flags, opt_ptr, protocol, header_len, checksum,
                                               source_port, dest_port, sequence_no, ack_no, window, payload_len)
            
        
            acked = False
            while not acked:
                try:
                    if self.CheckServer == True:
                        bytessent = self.sock.sendto((header + buffer), self.client_addr)
                    elif self.CheckServer == False:
                        bytessent = self.sock.sendto((header + buffer), self.serv_addr)
                    #print("%i byte payload sent. SEQNO = %d. Awaiting ACK..." % (len(buffer), sequence_no))
                    ack = self.sock.recvfrom(header_len)[0]
                    acked = True
                except syssock.timeout:
                    #print("Timeout occurred. Resending...")
                    pass

            # receive ACK
            self.last_pkt_recvd = struct.unpack('!BBBBHHLLQQLL', ack)
            if (self.last_pkt_recvd[1] == SOCK352_ACK) and (self.last_pkt_recvd[9] == sequence_no):
                #print("ACK received")
                pass
            else:
                #print("Response invalid")
                pass
            bytessent = bytessent - header_len

        else: # this is very easy when done with recursion but does it screw up the sequence numbers?
            payload_len = len(buffer) % MTU
            bytessent = bytessent + self.send(buffer[:payload_len])
            bytes_to_send = bytes_to_send - bytessent
            payload_len = MTU
            while bytes_to_send > 0:
                bytessent = bytessent + self.send(buffer[bytessent:bytessent+payload_len])
                bytes_to_send = bytes_to_send - payload_len
                

        if self.encrypt == True:
            bytessent -= 40

        return bytessent

    def recv(self,nbytes):
        if self.encrypt == True:
            nbytes += 40

        header_len = struct.calcsize('!BBBBHHLLQQLL')
        payload = ''
        payload_len = nbytes
        # receive packet
        if nbytes <= MTU:

            packet = self.sock.recvfrom(nbytes + header_len)[0]

            self.last_pkt_recvd = struct.unpack('!BBBBHHLLQQLL', packet[:header_len]) # read header
            payload = packet[header_len:header_len+self.last_pkt_recvd[11]]  # extract payload

            if self.encrypt == True:
                payload = self.box.decrypt(payload)
                opt_ptr = 0b1
            else:
                opt_ptr = 0b0

            # send ACK
            version = 0x1
            flags = SOCK352_ACK
            #opt_ptr = 0
            protocol = 0
            header_len = struct.calcsize('!BBBBHHLLQQLL')
            checksum = 0
            source_port = 0
            dest_port = 0
            sequence_no = self.last_pkt_recvd[8] + 1
            ack_no = self.last_pkt_recvd[8]
            window = 0
            
            #print("%i byte payload received. SEQNO = %d. Sending ACK..." % (payload_len, ack_no))
            
            payload_len = header_len
            
            ACK = self.udpPkt_hdr_data.pack(version, flags, opt_ptr, protocol, header_len, checksum,
                                               source_port, dest_port, sequence_no, ack_no, window, payload_len)
            if self.CheckServer == True:
                self.sock.sendto(ACK, self.client_addr)
            elif self.CheckServer == False:
                self.sock.sendto(ACK, self.serv_addr)
        else:
            bytes_to_recv = nbytes
            payload_len = nbytes % MTU
            if payload_len == 0:
                payload_len = MTU
            payload = payload + self.recv(payload_len)
            bytes_to_recv = bytes_to_recv - payload_len
            payload_len = MTU
            while bytes_to_recv > 0:
                payload = payload + self.recv(payload_len)
                bytes_to_recv = bytes_to_recv - payload_len 

        return payload



    

