SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  badges AS b,
  votes AS v,
  users AS u
WHERE ph.UserId = u.Id
  AND v.UserId = u.Id
  AND c.UserId = u.Id
  AND b.UserId = u.Id
  AND b.Date >= CAST('2010-09-26 12:17:14' AS timestamp)
  AND v.BountyAmount >= 0
  AND v.CreationDate >= CAST('2010-07-20 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-11 00:00:00' AS timestamp)
  AND u.DownVotes >= 0
  AND u.DownVotes <= 0
  AND u.UpVotes >= 0
  AND u.UpVotes <= 31
  AND u.CreationDate <= CAST('2014-08-06 20:38:52' AS timestamp);
