# frozen_string_literal: true

require_relative "dnslookup/version"
require 'optparse'
require 'open3'

module DNSLookup
  class Error < StandardError; end
  class CLI
    DEFAULT_SERVERS = ['8.8.8.8', '8.8.4.4'].freeze

    def self.run(argv: ARGV, out: $stdout)
      new(argv: argv, out: out).run
    end

    def initialize(argv: ARGV, out: $stdout)
      @argv = argv.dup
      @out = out
      @type = []
      @single_server = nil
    end

    def run
      parse_options
      domain = @argv.shift

      if domain.nil? || domain.start_with?('-')
        @out.puts "Error: You must specify a domain name.\n\n"
        @out.puts "Usage: dnslookup <domain name> [options]"
        exit 1
      end

      Query.new(domain: domain, types: @type, servers: query_servers, out: @out).run
    end

    def parse_options
      OptionParser.new do |opt|
        opt.banner = <<~BANNER
          Usage: dnslookup <domain name> [options]
          Example: dnslookup example.com -a -m -s8.8.8.8
        BANNER
        opt.on("-m", "--mx", "Return MX records") { @type << 'mx' }
        opt.on("-a", "--aname", "Return A name records") { @type << 'a' }
        opt.on("-c", "--cname", "Return C name records") { @type << 'c' }
        opt.on("-t", "--txt", "Return TXT records") { @type << 'txt' }
        opt.on("-s", "--server=SERVER", "Specify specific name server to query") do |v|
          @single_server = v
        end
        opt.on("-A", "--all", "Return all record types") { @type = %w[a mx c txt] }
        opt.on("-h", "--help", "Prints this help") do
          @out.puts opt
          exit
        end
        opt.on("-v", "--version", "Prints version") do
          @out.puts DNSLookup::VERSION
          exit
        end
      end.parse!(@argv)
    end

    def query_servers
      return [@single_server] if @single_server

      DEFAULT_SERVERS
    end
  end

  class Query
    RECORD_TYPES = {
      'a' => 'A',
      'mx' => 'MX',
      'c' => 'CNAME',
      'txt' => 'TXT'
    }.freeze

    def initialize(domain:, types: [], servers: CLI::DEFAULT_SERVERS, out: $stdout)
      @domain = domain
      @type = types
      @servers = servers
      @out = out
    end

    def run
      if @type.empty?
        query_command('a')
      else
        query_command(@type)
      end
    end

    def query_command(types)
      @servers.each do |server|
        Array(types).each do |type|
          record_type = normalize_record_type(type)
          check, error, status = Open3.capture3('dig', "@#{server}", @domain, record_type, '+short')
          @out.puts "Checking server: #{server} for #{record_type} records"
          @out.puts format_lookup_result(check: check, error: error, status: status)
          @out.puts
        end
      end
    rescue Errno::ENOENT
      @out.puts "(query failed: dig command not found)"
      @out.puts
    end

    private

    def normalize_record_type(type)
      RECORD_TYPES.fetch(type.to_s.downcase, type.to_s.upcase)
    end

    def format_lookup_result(check:, error:, status:)
      return check if status.success? && !check.empty?
      return "(no records found)" if status.success?

      error_message = error.to_s.strip
      return "(query failed)" if error_message.empty?

      "(query failed: #{error_message})"
    end
  end
end
