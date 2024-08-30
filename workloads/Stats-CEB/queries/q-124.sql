SELECT COUNT(*)
FROM postHistory AS ph,
  votes AS v,
  users AS u,
  badges AS b
WHERE u.Id = b.UserId
  AND u.Id = ph.UserId
  AND u.Id = v.UserId
  AND v.CreationDate <= CAST('2014-09-10 00:00:00' AS timestamp)
  AND u.DownVotes >= 0
  AND u.DownVotes <= 3
  AND u.UpVotes >= 0
  AND u.UpVotes <= 71
  AND b.Date >= CAST('2010-07-19 21:54:06' AS timestamp);
